import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

class DetCNN(nn.Module):
    def load_state_dict(self, state_dict, transfer = False, **kwargs):
        # Implement custom loader to make it possible to load a BNN checkpoint

        # Filter out the classification layers if transfer.
        if transfer:
            state_dict = {k:v for (k,v) in state_dict.items() if 'classify' not in k}

        return super().load_state_dict(state_dict, **kwargs)

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        new_shape = (x.shape[0], *self.shape)
        return x.view(new_shape)

class SimpleCNN(nn.Module):
    def __init__(self, nbr_classes, **kwargs):
        super(SimpleCNN, self).__init__()
        self.nbr_classes = nbr_classes
        self.cnn1 = nn.Conv3d(1, 32, 5, padding=2, stride=1)
        self.cnn2 = nn.Conv3d(32, nbr_classes, 1, padding=0, stride=1)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = F.Softplus(self.cnn1(x))
        x = self.cnn2(x)
        x = F.log_softmax(x, dim=1)
        return {'log_score': x}

class ConvBlock(nn.Module):
    def __init__(self, in_depth, out_depth, kernel_size, padding=0, stride=1, dilation = 1, dim = 3, bias=True):
        super(ConvBlock, self).__init__()
        conv = getattr(nn, 'Conv{}d'.format(dim))
        bn = getattr(nn, 'BatchNorm{}d'.format(dim))

        blocks = [conv(in_depth, out_depth, kernel_size, padding=padding, stride=stride, dilation = dilation, bias=bias)]
        blocks += [nn.Softplus()]
        blocks += [bn(out_depth)]
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)

class SkipAddBlock(nn.Module):
    def __init__(self, in_depth, out_depths, kernel_sizes, dilations, reduce_skip = False, dim = 3 ):
        super(SkipAddBlock, self).__init__()
        nbr_layers = len(out_depths)
        conv = getattr(nn, 'Conv{}d'.format(dim))
        bn = getattr(nn, 'BatchNorm{}d'.format(dim))
        assert nbr_layers > 1

        #Select paddings to keep output size equal to input size.
        paddings = [int(k_size/2)*dilation for k_size, dilation in zip(kernel_sizes, dilations)]

        self.first_pass = ConvBlock(in_depth, out_depths[0], kernel_sizes[0], padding = paddings[0], dilation=dilations[0], dim = dim)

        layers = []
        for i in range(1,nbr_layers-1):
            layers += [ConvBlock(out_depths[i-1], out_depths[i], kernel_sizes[i], padding = paddings[i], dilation=dilations[i], dim = dim)]
        layers += [conv(out_depths[-2], out_depths[-1], kernel_sizes[-1], padding = paddings[-1], dilation=dilations[-1])]
        self.second_pass = nn.Sequential(*layers)

        self.skip_connection = conv(out_depths[0], out_depths[-1], 1) if reduce_skip else nn.Identity()
        self.activation = nn.Sequential(nn.Softplus(), bn(out_depths[-1]))

    def forward(self, x):
        y1 = self.first_pass(x)
        y2 = self.second_pass(y1)
        y = y2 + self.skip_connection(y1)
        return self.activation(y)

class ReduceToSoftmax(nn.Module):
    def __init__(self, in_depth, out_depths, dim = 3):
        super(ReduceToSoftmax, self).__init__()
        nbr_layers = len(out_depths)
        conv = getattr(nn, 'Conv{}d'.format(dim))
        assert nbr_layers > 1
        layers = [ConvBlock(in_depth, out_depths[0], 1, dim = dim)]
        layers += [ConvBlock(out_depths[i-1], out_depths[i], 1, dim = dim) for i in range(1,nbr_layers-1)]
        layers += [conv(out_depths[-2], out_depths[-1], 1)]
        layers += [nn.LogSoftmax(dim=1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SSC_test(nn.Module):
    def __init__(self, nbr_classes, **kwargs):
        super(SSC_test, self).__init__()
        self.nbr_classes = nbr_classes
        self.base_depth = 8
        self.base_dilation = 2

        layers = [SkipAddBlock(1,out_depths = [self.base_depth, self.base_depth*2, self.base_depth*4],
                               kernel_sizes = [5,3,3],
                               dilations=3*[self.base_dilation],
                               reduce_skip = True)]
        layers += [ReduceToSoftmax(self.base_depth*4, out_depths=[self.base_depth*4, nbr_classes])]
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        y = self.layers(x)
        return {'log_score': y}

# Follows implementation in 'Semantic Scene Completion from a Single Depth Image' by Shuran Song Fisher Yu Andy Zeng Angel X. Chang Manolis Savva Thomas Funkhouser
# Uses dilation instead of pooling
class SSC(DetCNN):
    def __init__(self, nbr_classes, cfg):
        super(SSC, self).__init__()
        self.nbr_classes = nbr_classes
        self.base_depth = 8
        self.base_dilation = 1

        # Feature construction (green in article)
        layers = [SkipAddBlock(1,
                               out_depths = [self.base_depth, self.base_depth*2, self.base_depth*2],
                               kernel_sizes = [7,3,3],
                               dilations=3*[1],
                               reduce_skip = True)]
        #Dilated convolution instead of max pooling
        layers += [nn.Conv3d(self.base_depth*2, self.base_depth*2, 3,
                         dilation=2*self.base_dilation,
                         padding=2*self.base_dilation)]
        layers += [SkipAddBlock(self.base_depth*2,
                                out_depths = [self.base_depth*4, self.base_depth*4],
                                kernel_sizes = [3,3],
                                dilations=2*[self.base_dilation],
                                reduce_skip = True)]
        layers += [SkipAddBlock(self.base_depth*4,
                                out_depths = [self.base_depth*4, self.base_depth*4],
                                kernel_sizes = [3,3],
                                dilations=2*[self.base_dilation],
                                reduce_skip = False)]
        self.base_features = nn.Sequential(*layers)

        # Scale combining (yellow in article)
        layers = [SkipAddBlock(self.base_depth*4,
                                          out_depths = [self.base_depth*4, self.base_depth*4],
                                          kernel_sizes = [3,3],
                                          dilations=2*[self.base_dilation*2],
                                          reduce_skip = False)]
        layers += [SkipAddBlock(self.base_depth*4,
                                           out_depths = [self.base_depth*4, self.base_depth*4],
                                           kernel_sizes = [3,3],
                                           dilations=2*[self.base_dilation*2],
                                           reduce_skip = False)]
        self.scale_layers = nn.ModuleList(layers)

        # Classify (purple in article)
        input_depth = self.base_depth*4*(1+len(self.scale_layers))
        self.classify = ReduceToSoftmax(input_depth, out_depths=[self.base_depth*8, self.base_depth*8, nbr_classes])


    def forward(self, x):
        y_scales = [self.base_features(x)]
        for layer in self.scale_layers:
            y_scales.append(layer(y_scales[-1]))

        y_scales_cat = torch.cat(y_scales, dim = 1)
        y = self.classify(y_scales_cat)

        return {'log_score': y}

    def load_state_dict(self, state_dict, transfer = False, **kwargs):
        # Filter out the classification layers if transfer.
        if transfer:
            state_dict = {k:v for (k,v) in state_dict.items() if 'classify' not in k}

        return super().load_state_dict(state_dict, **kwargs)

class MNIST_CNN_simple(DetCNN):
    def __init__(self, nbr_classes, **kwargs):
        super().__init__()
        self.nbr_classes = nbr_classes
        self.img_size = (28,28)

        layers = [ConvBlock(1, 10, 3, padding = 1, dim = 2)]
        layers += [View([-1, 1])]
        layers += [ReduceToSoftmax(self.img_size[0]*self.img_size[1]*10, out_depths=[nbr_classes, nbr_classes], dim = 1)]
        layers += [View([-1])]
        self.layers = nn.Sequential(*layers)

    def forward(self,x):
        y = self.layers(x)
        return {'log_score': y}


class UNet(DetCNN):

    def __init__(self, nbr_classes, cfg, init_features=16):
        super().__init__()

        self.nbr_classes = nbr_classes
        self.cfg = cfg
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

        self.upconvs = [nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )]
        self.decoders = [self._block((features * 8) * 2, features * 8, name="dec4")]
        self.upconvs += [nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )]
        self.decoders += [self._block((features * 4) * 2, features * 4, name="dec3")]
        self.upconvs += [nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )]
        self.decoders += [self._block((features * 2) * 2, features * 2, name="dec2")]
        self.upconvs += [nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )]
        self.decoders += [self._block(features * 2, features, name="dec1")]

        self.upconvs = nn.ModuleList(self.upconvs)
        self.decoders = nn.ModuleList(self.decoders)

        self.crop = nn.ReplicationPad3d([-p for p in repl_pad]) if input_pad else nn.Identity()

        self.classify = ReduceToSoftmax(features, out_depths=[self.nbr_classes, self.nbr_classes])

    def forward(self, x):
        x = self.pad_input(x)
        encoded = []
        for i, (encode, pool) in enumerate(zip(self.encoders, self.pools)):
            input = x if i==0 else encoded[-1]
            e = encode(pool(input))
            encoded.append(e)

        d = encoded.pop() #bottleneck
        for e, decode, upconv in zip(encoded[::-1], self.decoders, self.upconvs):
            d = upconv(d)
            d = torch.cat((d, e), dim=1)
            d = decode(d)

        d = self.crop(d)
        y = self.classify(d)
        return {'log_score': y}

    def _block(self, in_channels, features, name):
        return nn.Sequential(
            ConvBlock(in_channels, features, 3, padding=1),
            ConvBlock(features, features, 3, padding=1)
            )
