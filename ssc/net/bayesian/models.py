import torch.nn as nn
import torch
# from utils.BBBlayers import BBBConv2d, BBBLinearFactorial, FlattenLayer
# from utils.conv2d import BBBConv2d
# from utils.linear import BBBLinearFactorial
import math
from collections import OrderedDict
from ssc.net.bayesian.layers import BReduceToProb, BSequential, BSkipAddBlock, BConvBlock, BConv, BConvTranspose, BMaxPool, BIdentity
from ssc.net.deterministic import View, ConvBlock
from ssc.net.bayesian.prior import Prior
import itertools


class BNN(nn.Module):
    def __init__(self, nbr_classes, cfg):
        super().__init__()
        self.bayesian = True
        self.fixed = False
        self.nbr_classes = nbr_classes
        self.prior = Prior.create_from_cfg(cfg)
        self.var_init = cfg['var_init']
        self.batch_norm = cfg['batch_norm']
        self.activation = cfg['activation']
        self.cfg = cfg

    def fix_parameters(self):
        for m in self.modules():
            m.fixed = True

    def release_parameters(self):
        for m in self.modules():
            m.fixed = False

    def forward(self,x):
        # Unless overriding this function the BNN needs to declare a BSequential module as self.layers.
        result = {}
        if self.fixed:
            result['log_score'] = self.layers(x)
            result['kl'] = 0
        else:
            result['log_score'], result['kl'] = self.layers(x)
        return result



class BSSC_test(BNN):
    def __init__(self, *args):
        super().__init__(*args)
        self.base_depth = 8
        self.base_dilation = 2
        self.output_dim = 3

        layers = [BSkipAddBlock(self.prior, self.var_init, 1,out_depths = [self.base_depth, self.base_depth*2, self.base_depth*4],
                               kernel_sizes = [5,3,3],
                               dilations=3*[self.base_dilation],
                               reduce_skip = True,
                               batch_norm = self.batch_norm,
                               activation = self.activation)]
        layers += [BReduceToProb(self.prior, self.var_init, self.base_depth*4,
                                 out_depths=[self.base_depth*4, self.nbr_classes],
                                 batch_norm = self.batch_norm,
                                 activation = self.activation)]
        self.layers = BSequential(*layers)

        self.release_parameters()



# Follows implementation in 'Semantic Scene Completion from a Single Depth Image' by Shuran Song Fisher Yu Andy Zeng Angel X. Chang Manolis Savva Thomas Funkhouser
# Uses dilation instead of pooling
class BSSC(BNN):
    def __init__(self, *args):
        super().__init__(*args)
        self.base_depth = 8 # 16 in paper
        self.base_dilation = 1
        self.output_dim = 3

        # Feature construction (green in article)
        layers = [BSkipAddBlock(self.prior, self.var_init, 1,
                               out_depths = [self.base_depth, self.base_depth*2, self.base_depth*2],
                               kernel_sizes = [7,3,3],
                               dilations=[1,1,1],
                               reduce_skip = True,
                               batch_norm = self.batch_norm,
                               activation = self.activation)]
        #Dilated convolution instead of max pooling
        layers += [BConv(self.prior, self.var_init, self.base_depth*2, self.base_depth*2, 3,
                         dilation=2*self.base_dilation,
                         padding=2*self.base_dilation)]
        layers += [BSkipAddBlock(self.prior, self.var_init, self.base_depth*2,
                                out_depths = [self.base_depth*4, self.base_depth*4],
                                kernel_sizes = [3,3],
                                dilations=2*[self.base_dilation],
                                reduce_skip = True,
                                batch_norm = self.batch_norm,
                                activation = self.activation)]
        layers += [BSkipAddBlock(self.prior, self.var_init, self.base_depth*4,
                                out_depths = [self.base_depth*4, self.base_depth*4],
                                kernel_sizes = [3,3],
                                dilations=2*[self.base_dilation],
                                reduce_skip = False,
                                batch_norm = self.batch_norm,
                                activation = self.activation)]
        self.base_features = BSequential(*layers)

        # Scale combining (yellow in article)
        layers = [BSkipAddBlock(self.prior, self.var_init, self.base_depth*4,
                                          out_depths = [self.base_depth*4, self.base_depth*4],
                                          kernel_sizes = [3,3],
                                          dilations=2*[self.base_dilation*2],
                                          reduce_skip = False,
                                          batch_norm = self.batch_norm,
                                          activation = self.activation)]
        layers += [BSkipAddBlock(self.prior, self.var_init, self.base_depth*4,
                                           out_depths = [self.base_depth*4, self.base_depth*4],
                                           kernel_sizes = [3,3],
                                           dilations=2*[self.base_dilation*2],
                                           reduce_skip = False,
                                           batch_norm = self.batch_norm,
                                           activation = self.activation)]
        self.scale_layers = nn.ModuleList(layers)

        # Classify (purple in article)
        input_depth = self.base_depth*4*(1+len(self.scale_layers))
        self.classify = BReduceToProb(self.prior, self.var_init, input_depth,
                                        out_depths=[self.base_depth*8, self.base_depth*8, self.nbr_classes],
                                        batch_norm = self.batch_norm,
                                        activation = self.activation)

        self.release_parameters()


    def forward(self, x):
        if self.fixed:
            return self.fixed_forward(x)

        y, acc_kl = self.base_features(x)
        y_scales = [y]
        for layer in self.scale_layers:
            y, kl = layer(y_scales[-1])
            acc_kl += kl
            y_scales.append(y)

        y_scales_cat = torch.cat(y_scales, dim = 1)
        y, kl = self.classify(y_scales_cat)

        acc_kl += kl

        return {'log_score': y, 'kl': acc_kl}

    def fixed_forward(self, x):
        result = {}
        y_scales = [self.base_features(x)]
        for layer in self.scale_layers:
            y_scales.append(layer(y_scales[-1]))

        y_scales_cat = torch.cat(y_scales, dim = 1)
        y = self.classify(y_scales_cat)

        return {'log_score': y}

class MNIST_BCNN(BNN):
    def __init__(self, *args):
        super().__init__(*args)
        self.base_depth = 8
        self.base_dilation = 2
        self.img_size = (28,28)
        self.output_dim = 1

        layers = [BSkipAddBlock(self.prior, self.var_init, 1,out_depths = [self.base_depth, self.base_depth*2, self.base_depth*4],
                               kernel_sizes = [5,3,3],
                               dilations=3*[self.base_dilation],
                               reduce_skip = True,
                               batch_norm = self.batch_norm,
                               dim = 2,
                               activation = self.activation)]
        layers += [View([-1, 1])]
        layers += [BReduceToProb(self.prior, self.var_init, self.img_size[0]*self.img_size[1]*self.base_depth*4,
                                 out_depths=[self.base_depth*4, self.nbr_classes],
                                 batch_norm = self.batch_norm,
                                 dim = self.output_dim,
                                 activation = self.activation)]
        layers += [View([-1])]
        self.layers = BSequential(*layers)

        self.release_parameters()


class MNIST_BCNN_simple(BNN):
    def __init__(self, *args):
        super().__init__(*args)
        self.img_size = (28,28)
        self.output_dim = 1

        layers = [BConvBlock(self.prior, self.var_init, 1, 10, 3,
                             padding = 1,
                             batch_norm = self.batch_norm,
                             dim = 2,
                             activation = self.activation)]
        layers += [View([-1, 1])]
        layers += [BReduceToProb(self.prior, self.var_init, self.img_size[0]*self.img_size[1]*10,
                                 out_depths=[self.nbr_classes, self.nbr_classes],
                                 batch_norm = self.batch_norm,
                                 dim = self.output_dim,
                                 activation = self.activation)]
        layers += [View([-1])]
        self.layers = BSequential(*layers)

        self.release_parameters()

class BSSC_UNet(BNN):

    def __init__(self, *args, init_features=16):
        super().__init__(*args)

        input_pad = self.cfg.get('UNet_padding', None)
        self.output_dim = 3

        features = init_features
        if input_pad:
            repl_pad = list(itertools.chain(*[(ip // 2, ip - (ip // 2)) for ip in input_pad]))
            self.pad_input = nn.ReplicationPad3d(repl_pad)
        else:
            self.pad_input = nn.Identity()
        self.pools = [nn.Identity()]
        self.encoders = [self._block(1, features, name="enc1")]
        self.pools += [nn.MaxPool3d(kernel_size=2, stride=2)]
        self.encoders += [self._block(features, features * 2, name="enc2")]
        self.pools += [nn.MaxPool3d(kernel_size=2, stride=2)]
        self.encoders += [self._block(features * 2, features * 4, name="enc3")]
        self.pools += [nn.MaxPool3d(kernel_size=2, stride=2)]
        self.encoders += [self._block(features * 4, features * 8, name="enc4")]
        self.pools += [nn.MaxPool3d(kernel_size=2, stride=2)]

        self.encoders += [self._block(features * 8, features * 16, name="bottleneck")]

        self.pools = nn.ModuleList(self.pools)
        self.encoders = nn.ModuleList(self.encoders)

        self.upconvs = [BConvTranspose(self.prior, self.var_init,
            features * 16, features * 8, kernel_size=2, stride=2, dim=3
        )]
        self.decoders = [self._block((features * 8) * 2, features * 8, name="dec4")]
        self.upconvs += [BConvTranspose(self.prior, self.var_init,
            features * 8, features * 4, kernel_size=2, stride=2, dim=3
        )]
        self.decoders += [self._block((features * 4) * 2, features * 4, name="dec3")]
        self.upconvs += [BConvTranspose(self.prior, self.var_init,
            features * 4, features * 2, kernel_size=2, stride=2, dim=3
        )]
        self.decoders += [self._block((features * 2) * 2, features * 2, name="dec2")]
        self.upconvs += [BConvTranspose(self.prior, self.var_init,
            features * 2, features, kernel_size=2, stride=2, dim=3
        )]
        self.decoders += [self._block(features * 2, features, name="dec1")]

        self.upconvs = nn.ModuleList(self.upconvs)
        self.decoders = nn.ModuleList(self.decoders)

        self.crop = nn.ReplicationPad3d([-p for p in repl_pad]) if input_pad else nn.Identity()

        self.classify = BReduceToProb(self.prior, self.var_init, features,
                                        out_depths=[self.nbr_classes, self.nbr_classes],
                                        batch_norm = self.batch_norm,
                                        activation = self.activation)

    def forward(self, x):
        x = self.pad_input(x)
        encoded = []
        acc_kl = 0
        for i, (encode, pool) in enumerate(zip(self.encoders, self.pools)):
            input = x if i==0 else encoded[-1]
            e, kl = encode(pool(input))
            encoded.append(e)
            acc_kl += kl

        d = encoded.pop() #bottleneck
        for e, decode, upconv in zip(encoded[::-1], self.decoders, self.upconvs):
            d, kl = upconv(d)
            acc_kl += kl
            d = torch.cat((d, e), dim=1)
            d, kl = decode(d)
            acc_kl += kl

        d = self.crop(d)
        y, kl = self.classify(d)
        acc_kl += kl
        return {'log_score': y, 'kl': acc_kl}

    def _block(self, in_channels, features, name):
        return BSequential(
            BConvBlock(self.prior, self.var_init, in_channels, features, 3, padding=1),
            BConvBlock(self.prior, self.var_init, features, features, 3, padding=1)
            )


class MetricRegression(nn.Module):
    def __init__(self, nbr_metrics, dim = 3):
        super().__init__()
        conv = getattr(nn,'Conv{}d'.format(dim))
        bn = getattr(nn, 'BatchNorm{}d'.format(dim))

        self.conv = nn.Sequential(
            ConvBlock(nbr_metrics, 2*nbr_metrics, 1, bias=False, dim=dim),
            conv(2*nbr_metrics, 1, 1, bias=False)
            )
        self.activation = nn.Sequential(
            # bn(1),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # Input (N, C, M, data_dim) where
        # N - Nbr Batches
        # C - Nbr Classes
        # M - Nbr Metrics

        N = x.shape[0]
        C = x.shape[1]
        M = x.shape[2]
        if x.ndim > 3:
            data_in_dim = data_out_dim = x.shape[3:]
        else:
            data_in_dim = [1]
            data_out_dim = []

        # Merge batch and class to one channel to enable 3D data input
        x = x.view([N*C, M, *data_in_dim])
        x = self.conv(x)
        x = x.view([N, C, *data_out_dim])
        x = self.activation(x)
        return x

class SampleNet(nn.Module):
    def __init__(self, bnn, cfg):
        super().__init__()
        assert bnn.bayesian
        self.bayesian = True
        self.fixed = False
        self.bnn = bnn
        self.metrics = cfg.get('fuse_metrics', None)


        if self.metrics:
            self.metrics.sort()
            self.sample_net = MetricRegression(len(self.metrics), bnn.output_dim)
            self.var_in_metrics = torch.any(torch.tensor(['var' in k for k in self.metrics]))
        else:
            self.var_in_metrics = False

        #Cfg
        self.nbr_f_samples = cfg['nbr_f_samples']
        self.freeze_bnn = cfg.get('freeze_bnn', False)

    def train(self, mode=True):
        super().train(mode)

        if self.freeze_bnn:
            self.bnn.eval()

    def fix_parameters(self):
        self.fixed = True
        self.bnn.fix_parameters()

    def release_parameters(self):
        self.fixed = False
        self.bnn.release_parameters()

    def _forward_bnn(self, x, no_grad = False):
        if no_grad:
            with torch.no_grad():
                y = self.bnn(x)
        else:
            y = self.bnn(x)

        return y

    def forward(self, x):
        # torch.autograd.set_detect_anomaly(True)
        if (
          self.fixed
          or self.nbr_f_samples < 2
          or (self.training and not self.metrics)
          ):
            return self._forward_bnn(x, no_grad=self.freeze_bnn or not self.training)

        # ------ Calculate stats over multiple forward passes ---------
        if self.training and not self.var_in_metrics:
            result = self._sample_entropy(x)
        else:
            result = self._sample_all(x)

        #------- Combine metrics for refined score -------
        if self.metrics:
            metric_cat = torch.cat([result[metric][:,:,None] for metric in self.metrics], dim=2)
            result['log_score'] = self.sample_net(metric_cat)
            result['fused_score'] = result['log_score'].exp()
        else:
            result['log_score'] = torch.log(result['mean_score'])

        return result

    def _sample_entropy(self, x):
        # Uses less memory if we don't need to compute the variance
        batchN = x.shape[0]
        data_dim = None
        bnn_no_grad = self.freeze_bnn or not self.training

        result = {}
        for i in range(self.nbr_f_samples):
            with torch.no_grad():
                bnn_result = self._forward_bnn(x, no_grad=(i+1 < self.nbr_f_samples) or bnn_no_grad)
            log_score = bnn_result['log_score']
            if not data_dim:
                data_dim = list(log_score.shape[1:])
                result['mean_score'] = mean_score = torch.zeros([batchN] + data_dim, device = x.device)
                result['entropy'] = entropy = torch.zeros([batchN] + data_dim, device = x.device)
            score = log_score.exp()
            mean_score += score
            entropy -= score*log_score

        entropy /= self.nbr_f_samples
        mean_score /= self.nbr_f_samples
        result['kl'] = bnn_result['kl']

        return result

    def _sample_all(self, x):
        batchN = x.shape[0]
        data_dim = None
        bnn_no_grad = self.freeze_bnn or not self.training

        # Start with entropy and mean score
        result = {}
        for i in range(self.nbr_f_samples):
            bnn_result = self._forward_bnn(x, no_grad=(i+1 < self.nbr_f_samples) or bnn_no_grad)
            log_score = bnn_result['log_score']
            if not data_dim:
                data_dim = list(log_score.shape[1:])
                scores = torch.zeros([batchN, self.nbr_f_samples] + data_dim, device = x.device)
                result['entropy'] = entropy = torch.zeros([batchN] + data_dim, device = x.device)
            scores[:,i] = score = log_score.exp()
            entropy -= score*log_score

        entropy /= self.nbr_f_samples
        result['mean_score'] = scores.mean(dim=1)
        result['kl'] = bnn_result['kl']

        # Calculate diagonal aleatoric and epistemic variances
        var_a = torch.zeros([batchN] + data_dim, device = x.device)
        var_e = torch.zeros_like(var_a, device = x.device)

        for si in range(self.nbr_f_samples):
            p = scores[:,si]
            var_a += p - p.pow(2)
            var_e += (p - result['mean_score']).pow(2)

        # Free memory for later
        del scores

        # Form the average
        var_a /= self.nbr_f_samples
        var_e /= self.nbr_f_samples

        result['var_aleatoric'] = var_a
        result['var_epistemic'] = var_e
        result['var'] = var_a + var_e

        if self.metrics:
            result['snr'] = result['mean_score']/(result['var']+1e-10)
            result['dispersion'] = result['var']/(result['mean_score'] + 1e-10)
            result['coef_var'] = torch.sqrt(result['var'] + 1e-10)/(result['mean_score'] + 1e-10)
            result['rel_entropy'] = result['entropy']/(result['mean_score'] + 1e-10)
            result['rel_entropy_inv'] = result['mean_score']/(result['entropy'] + 1e-10)

        return result

    def load_state_dict(self, state_dict, transfer = False):
        # Implement custom loader to make it possible to load a BNN checkpoint
        is_sample_net = False
        for k in state_dict.keys():
            if k.startswith('bnn'):
                is_sample_net = True
                break

        # Filter out the classification layers if transfer.
        if transfer:
            state_dict = {k:v for (k,v) in state_dict.items() if 'classify' not in k}
            state_dict = {k:v for (k,v) in state_dict.items() if 'sample_net' not in k}

        if is_sample_net:
            incompatible = super().load_state_dict(state_dict, strict = False)
        else:
            incompatible = self.bnn.load_state_dict(state_dict, strict = False)

        if incompatible.unexpected_keys:
            raise TypeError('Keys did not match, found unexpected keys: {}'.format(incompatible['unexpected_keys']))

        for k in incompatible.missing_keys:
            if 'sample_net' not in k:
                raise TypeError('Keys did not match, found missing keys: {}'.format(incompatible['missing_keys']))
