import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda
import chainer.links.model.vision.resnet as R
import collections
from chainer.functions.activation.relu import relu
from chainer.functions.activation.softmax import softmax
from chainer.functions.pooling.average_pooling_2d import average_pooling_2d
from chainer.initializers import normal
from chainer import link
from chainer.links.connection.convolution_2d import Convolution2D
from chainer.links.connection.linear import Linear
from chainer.links.normalization.batch_normalization import BatchNormalization


class BuildingBlock(link.Chain):

    def __init__(self, n_layer, in_channels, out_channels, stride, initialW=None,):
        super(BuildingBlock, self).__init__()
        with self.init_scope():
            self.a = BlockA(
                in_channels, out_channels, stride, initialW)
            self._forward = ["a"]
            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                block = BlockB(out_channels, initialW)
                setattr(self, name, block)
                self._forward.append(name)

    def __call__(self, x):
        for name in self._forward:
            l = getattr(self, name)
            x = l(x)
        return x

    @property
    def forward(self):
        return [getattr(self, name) for name in self._forward]


class BlockA(link.Chain):

    def __init__(self, in_channels, out_channels,
                 stride=2, initialW=None):
        super(BlockA, self).__init__()
        with self.init_scope():
            self.bn1 = BatchNormalization(in_channels)
            self.conv1 = Convolution2D(
                in_channels, out_channels, 3, stride, 1, initialW=initialW,
                nobias=True)
            self.bn2 = BatchNormalization(out_channels)
            self.conv2 = Convolution2D(
                out_channels, out_channels, 3, 1, 1, initialW=initialW,
                nobias=True)
            self.conv3 = Convolution2D(
                in_channels, out_channels, 1, stride, 0, initialW=initialW,
                nobias=True)

    def __call__(self, x):
        out = relu(self.bn1(x))
        h1 = self.conv1(out)
        h1 = self.conv2(relu(self.bn2(h1)))
        h2 = self.conv3(out)
        return h1 + h2


class BlockB(link.Chain):

    def __init__(self, in_channels, initialW=None):
        super(BlockB, self).__init__()
        with self.init_scope():
            self.bn1 = BatchNormalization(in_channels)
            self.conv1 = Convolution2D(
                in_channels, in_channels, 3, 1, 1, initialW=initialW,
                nobias=True)
            self.bn2 = BatchNormalization(in_channels)
            self.conv2 = Convolution2D(
                in_channels, in_channels, 3, 1, 1, initialW=initialW,
                nobias=True)

    def __call__(self, x):
        h = self.conv1(relu(self.bn1(x)))
        h = self.conv2(relu(self.bn2(h)))
        return h + x


class PreActResNet(chainer.Chain):

    def __init__(self, layer_names=None):
        super().__init__()
        kwargs = {'initialW': normal.HeNormal(scale=1.0)}

        block = [5, 5, 5]
        filters = [32, 32, 64, 128]

        with self.init_scope():
            self.conv1 = L.Convolution2D(None, filters[0], 3, 1, 1, **kwargs, nobias=True)
            self.res2 = BuildingBlock(block[0], filters[0], filters[1], 1, **kwargs)
            self.res3 = BuildingBlock(block[1], filters[1], filters[2], 2, **kwargs)
            self.res4 = BuildingBlock(block[2], filters[2], filters[3], 2, **kwargs)
            self.bn4 = L.BatchNormalization(filters[3])
            self.fc5 = L.Linear(filters[3], 10)

        self.functions = collections.OrderedDict([
            ('conv1', [self.conv1]),
            ('res2', [self.res2]),
            ('res3', [self.res3]),
            ('res4', [self.res4, self.bn4, F.relu]),
            ('pool4', [R._global_average_pooling_2d]),
            ('fc5', [self.fc5]),
        ])
        if layer_names is None:
            layer_names = list(self.functions.keys())[-1]
        if (not isinstance(layer_names, str) and
                all([isinstance(name, str) for name in layer_names])):
            return_tuple = True
        else:
            return_tuple = False
            layer_names = [layer_names]
        self._return_tuple = return_tuple
        self._layer_names = layer_names

    def __call__(self, x):
        h = x

        activations = dict()
        target_layers = set(self._layer_names)
        for key, funcs in self.functions.items():
            if len(target_layers) == 0:
                break
            for func in funcs:
                h = func(h)
            if key in target_layers:
                activations[key] = h
                target_layers.remove(key)

        if self._return_tuple:
            activations = tuple(
                [activations[name] for name in self._layer_names])
        else:
            activations = list(activations.values())[0]

        return activations
