# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" Inception-v3 model definition """

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.initializer import XavierUniform
import mindspore.ops as ops


class BasicConv2d(nn.Cell):
    """
    BasicConv2d
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, pad_mode='same', padding=0, has_bias=False):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                              pad_mode=pad_mode, padding=padding, weight_init=XavierUniform(), has_bias=has_bias)
        self.bn = nn.BatchNorm2d(out_channel, eps=0.001, momentum=0.9997)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionA(nn.Cell):
    """
    Inception A
    """
    def __init__(self, in_channels, pool_features, has_bias=False):
        super(InceptionA, self).__init__()
        self.concat = P.Concat(axis=1)
        self.branch0 = BasicConv2d(in_channels, 64, kernel_size=1, has_bias=has_bias)
        self.branch1 = nn.SequentialCell([
            BasicConv2d(in_channels, 48, kernel_size=1, has_bias=has_bias),
            BasicConv2d(48, 64, kernel_size=5, has_bias=has_bias)
        ])
        self.branch2 = nn.SequentialCell([
            BasicConv2d(in_channels, 64, kernel_size=1, has_bias=has_bias),
            BasicConv2d(64, 96, kernel_size=3, has_bias=has_bias),
            BasicConv2d(96, 96, kernel_size=3, has_bias=has_bias)

        ])
        self.branch_pool = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, pad_mode='same'),
            BasicConv2d(in_channels, pool_features, kernel_size=1, has_bias=has_bias)
        ])

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        branch_pool = self.branch_pool(x)
        out = self.concat((x0, x1, x2, branch_pool))
        return out


class InceptionB(nn.Cell):
    """
    Inception B
    """
    def __init__(self, in_channels, has_bias=False):
        super(InceptionB, self).__init__()
        self.concat = P.Concat(axis=1)
        self.branch0 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2, pad_mode='valid', has_bias=has_bias)
        self.branch1 = nn.SequentialCell([
            BasicConv2d(in_channels, 64, kernel_size=1, has_bias=has_bias),
            BasicConv2d(64, 96, kernel_size=3, has_bias=has_bias),
            BasicConv2d(96, 96, kernel_size=3, stride=2, pad_mode='valid', has_bias=has_bias)

        ])
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        branch_pool = self.branch_pool(x)
        out = self.concat((x0, x1, branch_pool))
        return out


class InceptionC(nn.Cell):
    """
    Inception C
    """
    def __init__(self, in_channels, channels_7x7, has_bias=False):
        super(InceptionC, self).__init__()
        self.concat = P.Concat(axis=1)
        self.branch0 = BasicConv2d(in_channels, 192, kernel_size=1, has_bias=has_bias)
        self.branch1 = nn.SequentialCell([
            BasicConv2d(in_channels, channels_7x7, kernel_size=1, has_bias=has_bias),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1, 7), has_bias=has_bias),
            BasicConv2d(channels_7x7, 192, kernel_size=(7, 1), has_bias=has_bias)
        ])
        self.branch2 = nn.SequentialCell([
            BasicConv2d(in_channels, channels_7x7, kernel_size=1, has_bias=has_bias),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(7, 1), has_bias=has_bias),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1, 7), has_bias=has_bias),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(7, 1), has_bias=has_bias),
            BasicConv2d(channels_7x7, 192, kernel_size=(1, 7), has_bias=has_bias)
        ])
        self.branch_pool = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, pad_mode='same'),
            BasicConv2d(in_channels, 192, kernel_size=1, has_bias=has_bias)
        ])

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        branch_pool = self.branch_pool(x)
        out = self.concat((x0, x1, x2, branch_pool))
        return out


class InceptionD(nn.Cell):
    """
    Inception D
    """
    def __init__(self, in_channels, has_bias=False):
        super(InceptionD, self).__init__()
        self.concat = P.Concat(axis=1)
        self.branch0 = nn.SequentialCell([
            BasicConv2d(in_channels, 192, kernel_size=1, has_bias=has_bias),
            BasicConv2d(192, 320, kernel_size=3, stride=2, pad_mode='valid', has_bias=has_bias)
        ])
        self.branch1 = nn.SequentialCell([
            BasicConv2d(in_channels, 192, kernel_size=1, has_bias=has_bias),
            BasicConv2d(192, 192, kernel_size=(1, 7), has_bias=has_bias),  # check
            BasicConv2d(192, 192, kernel_size=(7, 1), has_bias=has_bias),
            BasicConv2d(192, 192, kernel_size=3, stride=2, pad_mode='valid', has_bias=has_bias)
        ])
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        branch_pool = self.branch_pool(x)
        out = self.concat((x0, x1, branch_pool))
        return out


class InceptionEA(nn.Cell):
    """
    Inception E_1
    """
    def __init__(self, in_channels, has_bias=False):
        super(InceptionEA, self).__init__()
        self.concat = P.Concat(axis=1)
        self.branch0 = BasicConv2d(in_channels, 320, kernel_size=1, has_bias=has_bias)
        self.branch1 = BasicConv2d(in_channels, 384, kernel_size=1, has_bias=has_bias)
        self.branch1_a = BasicConv2d(384, 384, kernel_size=(1, 3), has_bias=has_bias)
        self.branch1_b = BasicConv2d(384, 384, kernel_size=(3, 1), has_bias=has_bias)
        self.branch2 = nn.SequentialCell([
            BasicConv2d(in_channels, 448, kernel_size=1, has_bias=has_bias),
            BasicConv2d(448, 384, kernel_size=3, has_bias=has_bias)
        ])
        self.branch2_a = BasicConv2d(384, 384, kernel_size=(1, 3), has_bias=has_bias)
        self.branch2_b = BasicConv2d(384, 384, kernel_size=(3, 1), has_bias=has_bias)
        self.branch_pool = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, pad_mode='same'),
            BasicConv2d(in_channels, 192, kernel_size=1, has_bias=has_bias)
        ])

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x1 = self.concat((self.branch1_a(x1), self.branch1_b(x1)))
        x2 = self.branch2(x)
        x2 = self.concat((self.branch2_a(x2), self.branch2_b(x2)))
        branch_pool = self.branch_pool(x)
        out = self.concat((x0, x1, x2, branch_pool))
        return out
class InceptionEB(nn.Cell):
    """
    Inception E_2
    """
    def __init__(self, in_channels, has_bias=False):
        super(InceptionEB, self).__init__()
        self.concat = P.Concat(axis=1)
        self.branch0 = BasicConv2d(in_channels, 320, kernel_size=1, has_bias=has_bias)
        self.branch1 = BasicConv2d(in_channels, 384, kernel_size=1, has_bias=has_bias)
        self.branch1_a = BasicConv2d(384, 384, kernel_size=(1, 3), has_bias=has_bias)
        self.branch1_b = BasicConv2d(384, 384, kernel_size=(3, 1), has_bias=has_bias)
        self.branch2 = nn.SequentialCell([
            BasicConv2d(in_channels, 448, kernel_size=1, has_bias=has_bias),
            BasicConv2d(448, 384, kernel_size=3, has_bias=has_bias)
        ])
        self.branch2_a = BasicConv2d(384, 384, kernel_size=(1, 3), has_bias=has_bias)
        self.branch2_b = BasicConv2d(384, 384, kernel_size=(3, 1), has_bias=has_bias)
        self.branch_pool = nn.SequentialCell([
            nn.MaxPool2d(kernel_size=3, pad_mode='same'),
            BasicConv2d(in_channels, 192, kernel_size=1, has_bias=has_bias)
        ])

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x1 = self.concat((self.branch1_a(x1), self.branch1_b(x1)))
        x2 = self.branch2(x)
        x2 = self.concat((self.branch2_a(x2), self.branch2_b(x2)))
        branch_pool = self.branch_pool(x)
        out = self.concat((x0, x1, x2, branch_pool))
        return out

class Logits(nn.Cell):
    """
    logits
    """
    def __init__(self, num_classes=10, dropout_keep_prob=0.8):
        super(Logits, self).__init__()
        self.avg_pool = nn.AvgPool2d(8, pad_mode='valid')
        self.dropout = nn.Dropout(p=1 - dropout_keep_prob)
        self.flatten = P.Flatten()
        self.fc = nn.Dense(2048, num_classes)

    def construct(self, x):
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class AuxLogits(nn.Cell):
    """
    AuxLogits
    """
    def __init__(self, in_channels, num_classes=10):
        super(AuxLogits, self).__init__()
        self.avg_pool = nn.AvgPool2d(5, stride=3, pad_mode='valid')
        self.conv2d_0 = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.conv2d_1 = nn.Conv2d(128, 768, kernel_size=5, pad_mode='valid')
        self.flatten = P.Flatten()
        self.fc = nn.Dense(in_channels, num_classes)

    def construct(self, x):
        x = self.avg_pool(x)
        x = self.conv2d_0(x)
        x = self.conv2d_1(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class InceptionV3(nn.Cell):
    """
    InceptionV3
    """
    def __init__(self, num_classes=10, is_training=False, has_bias=False, dropout_keep_prob=0.8, include_top=False):
        super(InceptionV3, self).__init__()
        self.is_training = is_training
        self.Conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2, pad_mode='valid', has_bias=has_bias)
        self.Conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1, pad_mode='valid', has_bias=has_bias)
        self.Conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, has_bias=has_bias)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b = BasicConv2d(64, 80, kernel_size=1, has_bias=has_bias)
        self.Conv2d_4a = BasicConv2d(80, 192, kernel_size=3, pad_mode='valid', has_bias=has_bias)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = InceptionA(192, pool_features=32, has_bias=has_bias)
        self.Mixed_5c = InceptionA(256, pool_features=64, has_bias=has_bias)
        self.Mixed_5d = InceptionA(288, pool_features=64, has_bias=has_bias)
        self.Mixed_6a = InceptionB(288, has_bias=has_bias)
        self.Mixed_6b = InceptionC(768, channels_7x7=128, has_bias=has_bias)
        self.Mixed_6c = InceptionC(768, channels_7x7=160, has_bias=has_bias)
        self.Mixed_6d = InceptionC(768, channels_7x7=160, has_bias=has_bias)
        self.Mixed_6e = InceptionC(768, channels_7x7=192, has_bias=has_bias)
        self.Mixed_7a = InceptionD(768, has_bias=has_bias)
        self.Mixed_7b = InceptionEA(1280, has_bias=has_bias)
        self.Mixed_7c = InceptionEB(2048, has_bias=has_bias)
        if is_training:
            self.aux_logits = AuxLogits(768, num_classes)
        self.include_top = include_top
        if self.include_top:
            self.logits = Logits(num_classes, dropout_keep_prob)
        self.resize = ops.interpolate
        self.reduceMean = ops.ReduceMean(keep_dims=True)
        self.squeeze_2 = ops.Squeeze(2)
        self.squeeze_3 = ops.Squeeze(3)

    def construct(self, x):
        """cell construct"""
        x = self.resize(x, size=(299, 299))
        x = 2 * x - 1
        x = self.Conv2d_1a(x)
        # print(x[0,:,0,0])
        x = self.Conv2d_2a(x)
        x = self.Conv2d_2b(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b(x)
        x = self.Conv2d_4a(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        if self.is_training:
            aux_logits = self.aux_logits(x)
        else:
            aux_logits = None
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        if not self.include_top:
            x = self.reduceMean(x, (2, 3))
            x = self.squeeze_2(self.squeeze_3(x))
            return x
        logits = self.logits(x)
        if self.is_training:
            return logits, aux_logits
        return logits
