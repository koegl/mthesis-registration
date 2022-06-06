# https://amaarora.github.io/2020/08/02/densenets.html
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Transtion(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transtion, self).__init__()
        # batch-norm
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))

        # 1x1 convolution down-samples from input features to output features
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        # growth rate: each layer adds K features on top of the previous layer
        # bn_size: the number of features in the bottleneck layer

        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),

        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        """ Bottleneck function """
        concatenated_features = torch.cat(inputs, 1)

        x = self.norm1(concatenated_features)
        x = self.relu1(x)

        bottleneck_output = self.conv1(x)

        return bottleneck_output

    def forward(self, input):
        if isinstance(input, torch.Tensor):
            prev_features = [input]
        else:
            prev_features = input

        bottleneck_output = self.bn_function(prev_features)

        x = self.norm2(bottleneck_output)
        x = self.relu2(x)

        new_features = self.conv2(x)

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()

        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]

        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)

        return torch.cat(features, 1)


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=24, memory_efficient=False):
        super(DenseNet, self).__init__()

        # First convolution and pooling
        # todo think of the input features for our 3D case
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(2, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Add multiple dense blocks and transition layers based on config
        # for densenet-121 config is [6, 12, 24, 16]
        num_features = num_init_features

        for i, num_layers in enumerate(block_config):
            # dense block
            block = _DenseBlock(
                num_layers,
                num_features,
                bn_size,
                growth_rate,
                drop_rate,
                memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)

            # each layer adds growth_rate amount of features
            num_features += num_layers*growth_rate

            # transition layer - if we are not at the end - it outputs half the features
            if i != len(block_config) - 1:
                transition = _Transtion(num_features, num_features // 2)
                self.features.add_module('transition%d' % (i + 1), transition)
                num_features = num_features // 2

        # final batch-norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        # linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # todo do I need this?
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x.float())

        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1)).view(features.size(0), -1)

        out = self.classifier(out)

        return out

