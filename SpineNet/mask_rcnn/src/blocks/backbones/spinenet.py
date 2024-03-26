# Copyright 2023 Huawei Technologies Co., Ltd
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
#
# This file or its part has been derived from the following repository
# and modified: https://github.com/yan-roo/SpineNet-Pytorch
# ============================================================================
from typing import ClassVar

import mindspore as ms
from mindspore import nn
from mindspore import ops

from ..layers import ConvModule, build_conv_layer, build_norm_layer
from ..initialization import constant_init, kaiming_init
from .. import Config


def prepare_layers_configs(conv_cfg=None, norm_cfg=None, act_cfg=None):
    if isinstance(conv_cfg, tuple):
        conv_cfg = dict(conv_cfg)
    elif isinstance(conv_cfg, Config):
        conv_cfg = conv_cfg.as_dict()

    if isinstance(norm_cfg, tuple):
        norm_cfg = dict(norm_cfg)
    elif isinstance(norm_cfg, Config):
        norm_cfg = norm_cfg.as_dict()
    return conv_cfg, norm_cfg, act_cfg


class BasicBlock(nn.Cell):
    """Basic ResNet block

    Args:
        inplanes (int): number of input channels
        planes (int): number of output channels
        stride (int): stride size in convolution
        dilation (int): dilation size in convolution
        downsample (ms.nn.SequentialCell): conv+bn block for residual
        norm_eval (bool): if True, BN layer will have only evaluation behaviour
        weights_update (bool): if False, all convolution layer will be frozen.
    """

    expansion = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            dilation: int = 1,
            downsample: nn.SequentialCell = None,
            norm_eval: bool = False,
            weights_update: bool = False,
            conv_cfg=None,
            norm_cfg=(('type', 'BN'),)
    ):
        super(BasicBlock, self).__init__()
        conv_cfg, norm_cfg, _ = prepare_layers_configs(
            conv_cfg=conv_cfg, norm_cfg=norm_cfg
        )

        self.weights_update = weights_update
        self.norm_eval = norm_eval
        self.affine = weights_update

        norm_cfg['affine'] = self.affine
        self.bn1 = build_norm_layer(norm_cfg, planes)
        self.bn2 = build_norm_layer(norm_cfg, planes)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            pad_mode='pad',
            dilation=dilation,
            has_bias=False
        )

        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, pad_mode='pad',
            has_bias=False
        )

        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

        if not weights_update:
            self.conv1.weight.requires_grad = False
            self.conv2.weight.requires_grad = False
            if self.downsample is not None:
                self.downsample[0].weight.requires_grad = False

        if self.norm_eval:
            self.bn1 = self.bn1.set_train(False)
            self.bn2 = self.bn2.set_train(False)
            if self.downsample is not None:
                self.downsample[1].set_train(False)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        """run forward"""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def set_train(self, mode: bool = True):
        """Set training mode"""
        super().set_train(mode=mode)
        self.bn1.set_train(mode and (not self.norm_eval))
        self.bn2.set_train(mode and (not self.norm_eval))
        if self.downsample is not None:
            self.downsample[1].set_train(
                mode and (not self.norm_eval)
            )


class Bottleneck(nn.Cell):
    """Bottleneck block for ResNet.

    Args:
        inplanes (int): number of input channels
        planes (int): number of output channels
        stride (int): stride size in convolution
        dilation (int): dilation size in convolution
        downsample (ms.nn.SequentialCell): conv+bn block for residual
        norm_eval (bool): if True, BN layer will have only evaluation behaviour
        weights_update (bool): if False, all convolution layer will be frozen.
    """

    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            dilation=1,
            downsample=None,
            norm_eval=False,
            weights_update=False,
            conv_cfg=None,
            norm_cfg=(('type', 'BN'),)
    ):
        super(Bottleneck, self).__init__()
        conv_cfg, norm_cfg, _ = prepare_layers_configs(
            conv_cfg=conv_cfg, norm_cfg=norm_cfg
        )
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.conv1_stride = 1
        self.conv2_stride = stride

        self.weights_update = weights_update
        self.norm_eval = norm_eval
        self.affine = weights_update
        norm_cfg['affine'] = self.affine
        self.bn1 = build_norm_layer(norm_cfg, planes)
        self.bn2 = build_norm_layer(norm_cfg, planes)
        self.bn3 = build_norm_layer(norm_cfg, planes * self.expansion)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            has_bias=False
        )
        self.conv2 = build_conv_layer(
            conv_cfg,
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            pad_mode='pad',
            dilation=dilation,
            has_bias=False
        )
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            has_bias=False
        )

        self.relu = nn.ReLU()
        self.downsample = downsample

        if not weights_update:
            self.conv1.weight.requires_grad = False
            self.conv2.weight.requires_grad = False
            self.conv3.weight.requires_grad = False
            if self.downsample is not None:
                self.downsample[0].weight.requires_grad = False

        if self.norm_eval:
            self.bn1 = self.bn1.set_train(False)
            self.bn2 = self.bn2.set_train(False)
            self.bn3 = self.bn3.set_train(False)
            if self.downsample is not None:
                self.downsample[1].set_train(False)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        """run forward"""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

    def set_train(self, mode: bool = True):
        """set train mode."""
        super(Bottleneck, self).set_train(mode=mode)
        self.bn1.set_train(mode and (not self.norm_eval))
        self.bn2.set_train(mode and (not self.norm_eval))
        self.bn3.set_train(mode and (not self.norm_eval))
        if self.downsample is not None:
            self.downsample[1].set_train(
                mode and (not self.norm_eval)
            )


def make_res_layer(
        block: ClassVar,
        inplanes: int,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilation: int = 1,
        weights_update: bool = True,
        norm_eval: bool = False,
        conv_cfg=None,
        norm_cfg=(('type', 'BN'),)
) -> nn.Cell:
    """Make residual block."""
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.SequentialCell(
            build_conv_layer(
                conv_cfg,
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                has_bias=False
            ),
            build_norm_layer(norm_cfg, planes * block.expansion)
        )

    layers = [block(
        inplanes,
        planes,
        stride,
        dilation,
        downsample,
        norm_eval=norm_eval,
        weights_update=weights_update,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg
    )]
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(
            block(
                inplanes,
                planes,
                1,
                dilation,
                norm_eval=norm_eval,
                weights_update=weights_update,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg
            )
        )

    return nn.SequentialCell(layers)


FILTER_SIZE_MAP = {
    1: 32,
    2: 64,
    3: 128,
    4: 256,
    5: 256,
    6: 256,
    7: 256,
}

# The fixed SpineNet architecture discovered by NAS.
# Each element represents a specification of a building block:
#   (block_level, block_fn, (input_offset0, input_offset1), is_output).
SPINENET_BLOCK_SPECS = [
    (2, Bottleneck, (None, None), False),  # init block
    (2, Bottleneck, (None, None), False),  # init block
    (2, Bottleneck, (0, 1), False),
    (4, BasicBlock, (0, 1), False),
    (3, Bottleneck, (2, 3), False),
    (4, Bottleneck, (2, 4), False),
    (6, BasicBlock, (3, 5), False),
    (4, Bottleneck, (3, 5), False),
    (5, BasicBlock, (6, 7), False),
    (7, BasicBlock, (6, 8), False),
    (5, Bottleneck, (8, 9), False),
    (5, Bottleneck, (8, 10), False),
    (4, Bottleneck, (5, 10), True),
    (3, Bottleneck, (4, 10), True),
    (5, Bottleneck, (7, 12), True),
    (7, Bottleneck, (5, 14), True),
    (6, Bottleneck, (12, 14), True),
]

SCALING_MAP = {
    '49S': {
        'endpoints_num_filters': 128,
        'filter_size_scale': 0.65,
        'resample_alpha': 0.5,
        'block_repeats': 1,
    },
    '49': {
        'endpoints_num_filters': 256,
        'filter_size_scale': 1.0,
        'resample_alpha': 0.5,
        'block_repeats': 1,
    },
    '96': {
        'endpoints_num_filters': 256,
        'filter_size_scale': 1.0,
        'resample_alpha': 0.5,
        'block_repeats': 2,
    },
    '143': {
        'endpoints_num_filters': 256,
        'filter_size_scale': 1.0,
        'resample_alpha': 1.0,
        'block_repeats': 3,
    },
    '190': {
        'endpoints_num_filters': 512,
        'filter_size_scale': 1.3,
        'resample_alpha': 1.0,
        'block_repeats': 4,
    },
}

class BlockSpec:
    """A container class that specifies the block configuration for SpineNet."""

    def __init__(self, level, block_fn, input_offsets, is_output):
        self.level = level
        self.block_fn = block_fn
        self.input_offsets = input_offsets
        self.is_output = is_output


def build_block_specs(block_specs=None):
    """Builds the list of BlockSpec objects for SpineNet."""
    if not block_specs:
        block_specs = SPINENET_BLOCK_SPECS
    return [BlockSpec(*b) for b in block_specs]


class Resample(nn.Cell):
    def __init__(
            self,
            in_channels,
            out_channels,
            scale,
            block_type,
            input_size,
            norm_cfg=Config(dict(type="BN")),
            alpha=1.0,

    ):
        super(Resample, self).__init__()
        _, norm_cfg, _ = prepare_layers_configs(norm_cfg=norm_cfg)
        self.scale = scale
        new_in_channels = int(in_channels * alpha)
        if block_type == Bottleneck:
            in_channels *= 4
        self.squeeze_conv = ConvModule(
            in_channels, new_in_channels, 1, norm_cfg=norm_cfg
        )
        self.output_size = tuple([int(s * self.scale) for s in input_size])

        if scale == 1:
            self.resize_op = nn.Identity()
        elif scale > 1:
            self.resize_op = ops.ResizeNearestNeighbor(size=self.output_size)
        if scale < 1:
            self.downsample_conv = ConvModule(
                new_in_channels, new_in_channels, 3, padding=1, stride=2,
                norm_cfg=norm_cfg
            )
            if self.scale < 0.5:
                new_kernel_size = 3 if self.scale >= 0.25 else 5
                self.max_pool = nn.SequentialCell(
                    nn.Pad(
                        ((0, 0), (0, 0),
                         (new_kernel_size // 2, new_kernel_size // 2),
                         (new_kernel_size // 2, new_kernel_size // 2))
                    ),
                    nn.MaxPool2d(
                        kernel_size=3 if self.scale >= 0.25 else 5,
                        stride=int(0.5 / self.scale)
                    )
                )

        self.expand_conv = ConvModule(
            new_in_channels, out_channels, 1, norm_cfg=norm_cfg,
            act_cfg=None
        )

    def _resize(self, x):
        if self.scale >= 1:
            x = self.resize_op(x)
        if self.scale < 1:
            x = self.downsample_conv(x)
            if self.scale < 0.5:
                x = self.max_pool(x)
        return x

    def construct(self, inputs):
        feat = self.squeeze_conv(inputs)
        feat = self._resize(feat)
        feat = self.expand_conv(feat)
        return feat


class Merge(nn.Cell):
    """Merge two input tensors"""

    def __init__(
            self, block_spec, norm_cfg, alpha, filter_size_scale, input_sizes
    ):
        super(Merge, self).__init__()
        _, norm_cfg, _ = prepare_layers_configs(norm_cfg=norm_cfg)

        out_channels = int(
            FILTER_SIZE_MAP[block_spec.level] * filter_size_scale
        )
        if block_spec.block_fn == Bottleneck:
            out_channels *= 4
        self.block = block_spec.block_fn

        resample_ops = []
        self.input_sizes = input_sizes
        self.output_size = None
        for i, spec_idx in enumerate(block_spec.input_offsets):
            spec = BlockSpec(*SPINENET_BLOCK_SPECS[spec_idx])
            in_channels = int(FILTER_SIZE_MAP[spec.level] * filter_size_scale)
            scale = 2 ** (spec.level - block_spec.level)
            resample = Resample(
                in_channels=in_channels,
                out_channels=out_channels,
                scale=scale,
                block_type=spec.block_fn,
                input_size=self.input_sizes[i],
                norm_cfg=norm_cfg,
                alpha=alpha
            )
            resample_ops.append(resample)
            self.output_size = resample_ops[-1].output_size

        self.resample_ops = nn.CellList(resample_ops)

    def construct(self, inputs):
        assert len(inputs) == len(self.resample_ops)
        parent0_feat = self.resample_ops[0](inputs[0])
        parent1_feat = self.resample_ops[1](inputs[1])
        target_feat = parent0_feat + parent1_feat
        return target_feat


class SpineNet(nn.Cell):
    """Class to build SpineNet backbone"""

    def __init__(
            self,
            arch,
            img_size,
            in_channels=3,
            output_level=(3, 4, 5, 6, 7),
            conv_cfg=None,
            norm_cfg=(('type', 'BN'), ('requires_grad', True)),
            zero_init_residual=True
    ):
        super(SpineNet, self).__init__()
        conv_cfg, norm_cfg, _ = prepare_layers_configs(
            conv_cfg=conv_cfg, norm_cfg=norm_cfg
        )
        self._block_specs = build_block_specs()[2:]
        self._endpoints_num_filters = SCALING_MAP[arch]['endpoints_num_filters']
        self._resample_alpha = SCALING_MAP[arch]['resample_alpha']
        self._block_repeats = SCALING_MAP[arch]['block_repeats']
        self._filter_size_scale = SCALING_MAP[arch]['filter_size_scale']
        self._init_block_fn = Bottleneck
        self._num_init_blocks = 2
        self.h, self.w = img_size
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.zero_init_residual = zero_init_residual
        assert min(output_level) > 2 and max(
            output_level) < 8, "Output level out of range"
        self.output_level = output_level

        self.sizes = [(self.h, self.w)]

        self.spec_levels = [spec.level for spec in self._block_specs]
        self.spec_input_offsets = [spec.input_offsets for spec in
                                   self._block_specs]
        self.spec_is_output = [spec.is_output for spec in self._block_specs]
        self.spec_num = len(self._block_specs)

        self._make_stem_layer(in_channels)
        self._make_scale_permuted_network()
        self._make_endpoints()

    def _make_stem_layer(self, in_channels):
        """Build the stem network."""
        # Build the first conv and maxpooling layers.
        # divide size by 2
        self.conv1 = ConvModule(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg
        )
        self.maxpool = nn.SequentialCell(
            nn.Pad(((0, 0), (0, 0), (1, 1), (1, 1))),
            nn.MaxPool2d(
                kernel_size=3, stride=2, pad_mode='valid'
            )
        )

        # Build the initial level 2 blocks.
        self.init_block1 = make_res_layer(
            self._init_block_fn,
            64,
            int(FILTER_SIZE_MAP[2] * self._filter_size_scale),
            self._block_repeats,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)
        self.init_block2 = make_res_layer(
            self._init_block_fn,
            int(FILTER_SIZE_MAP[2] * self._filter_size_scale) * 4,
            int(FILTER_SIZE_MAP[2] * self._filter_size_scale),
            self._block_repeats,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)

        self.sizes.append(
            (self.sizes[-1][0] // 4, self.sizes[-1][1] // 4)
        )

    def _make_endpoints(self):
        endpoint_convs = [nn.Identity() for _ in range(self.spec_num)]
        for block_spec in self._block_specs:
            if block_spec.is_output:
                in_channels = int(
                    FILTER_SIZE_MAP[block_spec.level] * self._filter_size_scale
                ) * 4
                endpoint_convs[block_spec.level] = ConvModule(
                    in_channels,
                    self._endpoints_num_filters,
                    kernel_size=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=None
                )
        self.endpoint_convs = nn.CellList(endpoint_convs)

    def _make_scale_permuted_network(self):
        merge_ops = []
        scale_permuted_blocks = []

        for spec in self._block_specs:
            merge_ops.append(
                Merge(
                    block_spec=spec, norm_cfg=self.norm_cfg,
                    alpha=self._resample_alpha,
                    filter_size_scale=self._filter_size_scale,
                    input_sizes=[
                        self.sizes[spec.input_offsets[0]],
                        self.sizes[spec.input_offsets[1]]
                    ]
                )
            )
            channels = int(
                FILTER_SIZE_MAP[spec.level] * self._filter_size_scale)
            in_channels = (
                channels * 4 if spec.block_fn == Bottleneck else channels
            )
            scale_permuted_blocks.append(
                make_res_layer(
                    spec.block_fn,
                    in_channels,
                    channels,
                    self._block_repeats,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg
                )
            )
            self.sizes.append(merge_ops[-1].output_size)

        self.merge_ops = nn.CellList(merge_ops)
        self.scale_permuted_blocks = nn.CellList(scale_permuted_blocks)

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                constant_init(m, 1)
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    constant_init(m.norm3, 0)
                elif isinstance(m, BasicBlock):
                    constant_init(m.norm2, 0)

    def construct(self, x):

        feat = self.maxpool(self.conv1(x))
        feat1 = self.init_block1(feat)
        feat2 = self.init_block2(feat1)
        block_feats = [feat1, feat2]

        num_outgoing_connections = [0, 0]
        output_feat = [None for _ in range(self.spec_num)]

        for i in range(self.spec_num):
            target_feat = self.merge_ops[i](
                [block_feats[feat_idx] for feat_idx in self.spec_input_offsets[i]]
            )
            # Connect intermediate blocks with outdegree 0 to the output block.
            if self.spec_is_output[i]:
                for j, (j_feat, j_connections) in enumerate(
                        zip(block_feats, num_outgoing_connections)):
                    if j_connections == 0 and j_feat.shape == target_feat.shape:
                        target_feat += j_feat
                        num_outgoing_connections[j] += 1
            target_feat = ops.relu(target_feat)
            target_feat = self.scale_permuted_blocks[i](target_feat)
            block_feats.append(target_feat)
            num_outgoing_connections.append(0)
            for feat_idx in self.spec_input_offsets[i]:
                num_outgoing_connections[feat_idx] += 1
            if self.spec_is_output:
                output_feat[self.spec_levels[i]] = target_feat

        return [self.endpoint_convs[level](output_feat[level]) for level in self.output_level]
