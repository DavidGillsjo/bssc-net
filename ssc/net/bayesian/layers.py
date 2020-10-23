import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


def get_activation(name):
    name = name.lower().split('-')[0]
    if name == 'relu':
        return nn.ReLU
    elif name == 'softplus':
        return nn.Softplus
    else:
        raise NameError('No activation named {}'.format(name))

def get_final_activation(name):
    name = name.lower().split('-')[1]
    if name == 'softmax':
        return nn.LogSoftmax
    elif name == 'softplus':
        return LogNormalizedSoftPlus


class LogNormalizedSoftPlus(nn.Softplus):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = super().forward(x)
        x_n = torch.log(x) - torch.log(x.sum(dim=self.dim, keepdims=True))
        return x_n

class BNLLLoss(nn.NLLLoss):
    def __init__(self, beta, nbr_batches, nbr_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nbr_weights = float(nbr_weights)

        if not hasattr(beta, 'strip'):
            self.beta_f = lambda b_idx : beta
        elif beta.lower() == 'blundell':
            M = nbr_batches
            self.beta_f = lambda b_idx: (2**(M - b_idx)) / (2**M - 1)
        else:
            raise NotImplementedError('{} not supported'.format(beta))

    def forward(self, scores, kl, target, batch_idx):
        class_loss = F.nll_loss(scores, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
        return class_loss + self.beta_f(batch_idx)*kl/self.nbr_weights


class BSequential(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.layers = nn.ModuleList(modules)
        self.bayesian = True
        self.fixed = False

    def forward(self, x):
        if self.fixed:
            for layer in self.layers:
                x = layer(x)
            return x

        kl = 0

        for layer in self.layers:
            if getattr(layer, 'bayesian', False):
                x, kl_ = layer(x)
                kl += kl_
            else:
                x = layer(x)

        return x, kl

class BMaxPool(nn.Module):
    def __init__(self, kernel_size, dim=3, **kwargs):
        super().__init__()
        self.bayesian = True
        self.fixed = False
        mpool_mod = getattr(nn, 'MaxPool{}d'.format(dim))
        self.mpool = mpool_mod(kernel_size, **kwargs)

    def forward(self, x):
        y = self.mpool(x)
        return y if self.fixed else (y,0)

class BConv(nn.Module):
    def __init__(self, prior, var_init, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, dim = 3, transposed=False):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.prior = prior
        self.var_init = var_init
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bayesian = True
        self.fixed = False
        self.dim = dim
        self.transposed = transposed
        self.conv = getattr(F, 'conv{}d'.format(dim))

        if bias:
            self.mu_bias = Parameter(torch.Tensor(out_channels))
            self.sigma_bias = Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels))
        else:
            self.register_parameter('mu_bias', None)

        if self.transposed:
            tensor_dim = [in_channels, out_channels // groups] + [kernel_size]*self.dim
        else:
            tensor_dim = [out_channels, in_channels // groups] + [kernel_size]*self.dim
        self.mu_weight = Parameter(torch.Tensor(*tensor_dim))
        self.sigma_weight = Parameter(torch.Tensor(*tensor_dim))
        self.register_buffer('eps_weight', torch.Tensor(*tensor_dim))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        n *= self.kernel_size ** self.dim
        stdv = 1.0 / math.sqrt(n)
        self.mu_weight.data.uniform_(-stdv, stdv)
        self.sigma_weight.data.fill_(math.log(self.var_init))
        if self.mu_bias is not None:
            self.mu_bias.data.uniform_(-stdv, stdv)
            self.sigma_bias.data.fill_(math.log(self.var_init))

    def _sample_w(self, calc_kl = True):
        sig_weight = torch.exp(self.sigma_weight)
        weight = self.mu_weight + sig_weight * self.eps_weight.normal_()
        kl = self.prior.kl_div(weight, self.mu_weight, sig_weight) if calc_kl else 0
        return sig_weight, weight, kl

    def _sample_bias(self, calc_kl = True):
        if self.mu_bias is None:
            bias = None
            kl = 0
        else:
            sig_bias = torch.exp(self.sigma_bias)
            bias = self.mu_bias + sig_bias * self.eps_bias.normal_()
            kl = self.prior.kl_div(bias, self.mu_bias, sig_bias) if calc_kl else 0

        return bias, kl

    def sample_weights(self):
        sig_weight, weight, kl = self._sample_w(calc_kl = False)
        return weight

    def forward(self, input):
        # Special case, used for tracing the graph
        if self.fixed:
            return self.conv(input, self.mu_weight, self.mu_bias, self.stride, self.padding, self.dilation, self.groups)

        sig_weight, weight, kl = self._sample_w()
        bias, bias_kl = self._sample_bias()

        out = self.conv(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return out, kl + bias_kl

class BConvTranspose(BConv):
    def __init__(self, *args, output_padding = 0, **kwargs):
        super().__init__(*args, transposed=True, **kwargs)
        self.output_padding = output_padding
        self.conv = getattr(F, 'conv_transpose{}d'.format(self.dim))

    def forward(self, x, output_size=None):
        assert output_size is None, 'output padding function not implemented'
        output_padding = nn.modules.utils._single(self.output_padding)

        if self.fixed:
            return self.conv(x, self.mu_weight, self.mu_bias, self.stride, self.padding, output_padding, self.groups, self.dilation)

        sig_weight, weight, kl = self._sample_w()
        bias, bias_kl = self._sample_bias()

        out = self.conv(x, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)
        return out, kl + bias_kl

class BConvBlock(nn.Module):
    def __init__(self, prior, var_init, in_depth, out_depth, kernel_size, padding=0, stride=1, dilation = 1, batch_norm = True, dim = 3, activation = 'softplus'):
        super().__init__()
        bn = getattr(nn, 'BatchNorm{}d'.format(dim))
        act_fn = get_activation(activation)
        self.bayesian = True
        blocks = [BConv(prior, var_init, in_depth, out_depth, kernel_size, padding=padding, stride=stride, dilation = dilation, dim = dim)]
        blocks += [act_fn()]
        if batch_norm:
            blocks += [bn(out_depth)]
        self.block = BSequential(*blocks)

    def forward(self, x):
        return self.block(x)

class BIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.bayesian = True
        self.fixed = False

    def forward(self, x):
        return x if self.fixed else (x, 0)

class BSkipAddBlock(nn.Module):
    def __init__(self, prior, var_init, in_depth, out_depths, kernel_sizes, dilations, reduce_skip = False, batch_norm = True, dim = 3, activation = 'softplus' ):
        super().__init__()
        self.bayesian = True
        self.fixed = False
        nbr_layers = len(out_depths)
        bn = getattr(nn, 'BatchNorm{}d'.format(dim))
        act_fn = get_activation(activation)
        assert nbr_layers > 1

        #Select paddings to keep output size equal to input size.
        paddings = [int(k_size/2)*dilation for k_size, dilation in zip(kernel_sizes, dilations)]

        self.first_pass = BConvBlock(prior, var_init, in_depth, out_depths[0], kernel_sizes[0], padding = paddings[0], dilation=dilations[0], batch_norm = batch_norm, dim = dim, activation = activation)

        layers = []
        for i in range(1,nbr_layers-1):
            layers += [BConvBlock(prior, var_init, out_depths[i-1], out_depths[i], kernel_sizes[i], padding = paddings[i], dilation=dilations[i], batch_norm = batch_norm, dim = dim, activation = activation)]
        layers += [BConv(prior, var_init, out_depths[-2], out_depths[-1], kernel_sizes[-1], padding = paddings[-1], dilation=dilations[-1], dim = dim)]
        self.second_pass = BSequential(*layers)

        self.skip_connection = BConv(prior, var_init, out_depths[0], out_depths[-1], 1, dim = dim) if reduce_skip else BIdentity()
        if batch_norm:
            self.activation = nn.Sequential(act_fn(), bn(out_depths[-1]))
        else:
            self.activation = act_fn

    def forward(self, x):
        return self.fixed_forward(x) if self.fixed else self.sampled_forward(x)

    def sampled_forward(self, x):
        y1, kl1 = self.first_pass(x)
        y2, kl2 = self.second_pass(y1)
        y3, kl3 = self.skip_connection(y1)
        y = y2 + y3
        kl = kl1 + kl2 + kl3
        return self.activation(y), kl

    def fixed_forward(self, x):
        y1 = self.first_pass(x)
        y2 = self.second_pass(y1)
        y3 = self.skip_connection(y1)
        y = y2 + y3
        return self.activation(y)

class BReduceToProb(nn.Module):
    def __init__(self, prior, var_init, in_depth, out_depths, batch_norm = True, dim = 3, activation = 'softplus'):
        super().__init__()
        self.bayesian = True
        self.fixed = False
        nbr_layers = len(out_depths)
        final_act_fn = get_final_activation(activation)
        assert nbr_layers > 1
        layers = [BConvBlock(prior, var_init, in_depth, out_depths[0], 1, batch_norm = batch_norm, dim = dim, activation = activation)]
        layers += [BConvBlock(prior, var_init, out_depths[i-1], out_depths[i], 1, batch_norm = batch_norm, dim = dim, activation = activation) for i in range(1,nbr_layers-1)]
        layers += [BConv(prior, var_init, out_depths[-2], out_depths[-1], 1, dim = dim)]
        layers += [final_act_fn(dim=1)]
        self.layers = BSequential(*layers)

    def forward(self, input):
        return self.layers(input)
