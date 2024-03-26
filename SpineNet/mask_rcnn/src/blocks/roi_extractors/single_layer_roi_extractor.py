# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
"""Legacy single level ROI extractor module."""
from typing import List

import numpy as np
import mindspore as ms

from mindspore import ops
from mindspore import nn
from mindspore.nn import layer as L
from mindspore.common.tensor import Tensor

from .. import Config


class ROIAlign(nn.Cell):
    """
    Extract RoI features from multiple feature map.

    Args:
        out_size_h (int) - RoI height.
        out_size_w (int) - RoI width.
        spatial_scale (int) - RoI spatial scale.
        sample_num (int) - RoI sample number.
    """
    def __init__(self,
                 out_size_h: int,
                 out_size_w: int,
                 spatial_scale: int,
                 sample_num: int = 0):
        super(ROIAlign, self).__init__()

        self.out_size = (out_size_h, out_size_w)
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
        self.align_op = ops.ROIAlign(self.out_size[0], self.out_size[1],
                                     self.spatial_scale, self.sample_num)

    def construct(self, features: ms.Tensor, rois: ms.Tensor):
        return self.align_op(features, rois)

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += f'(out_size={self.out_size}, ' \
                      f'spatial_scale={self.spatial_scale}, ' \
                      f'sample_num={self.sample_num}'
        return format_str


class SingleRoIExtractor(nn.Cell):
    """
    Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (Config): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (List[int]): Strides of input feature maps.
        train_batch_size (int): number of train rois.
        test_batch_size (int): number of validation rois.
        finest_scale (int): Scale threshold of mapping to level 0.
    """

    def __init__(self,
                 roi_layer: Config,
                 out_channels: int,
                 featmap_strides: List[int],
                 train_batch_size: int = 1,
                 test_batch_size: int = 1,
                 finest_scale: int = 56):
        super(SingleRoIExtractor, self).__init__()
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.out_size = roi_layer.out_size
        self.sample_num = roi_layer.sample_num
        self.roi_layers = self.build_roi_layers(self.featmap_strides)
        self.roi_layers = L.CellList(self.roi_layers)

        self.sqrt = ops.Sqrt()
        self.log = ops.Log()
        self.finest_scale_ = finest_scale
        self.clamp = ops.clip_by_value

        self.cast = ops.Cast()
        self.equal = ops.Equal()
        self.select = ops.Select()

        self.dtype = np.float32
        self.ms_dtype = ms.float32

        # training mode depended variables
        self.training_local = True
        self.batch_size = None
        self.ones = None
        self.finest_scale = None
        self.epslion = None
        self.zeros = None
        self.max_levels = None
        self.twos = None
        self.res_ = None

        self.set_train_local(training=self.training_local)

    def set_train_local(self, training=True):
        """Set training flag."""
        self.training_local = training

        # Init tensor
        self.batch_size = (
            self.train_batch_size if self.training_local
            else self.test_batch_size
        )
        self.ones = Tensor(
            np.array(np.ones((self.batch_size, 1)), dtype=self.dtype)
        )
        finest_scale = np.array(
            np.ones((self.batch_size, 1)), dtype=self.dtype
        ) * self.finest_scale_
        self.finest_scale = Tensor(finest_scale)
        self.epslion = Tensor(
            np.array(np.ones((self.batch_size, 1)), dtype=self.dtype) *
            self.dtype(1e-6)
        )
        self.zeros = Tensor(
            np.array(np.zeros((self.batch_size, 1)), dtype=np.int32)
        )
        self.max_levels = Tensor(
            np.array(np.ones((self.batch_size, 1)), dtype=np.int32) *
            (self.num_levels-1)
        )
        self.twos = Tensor(
            np.array(np.ones((self.batch_size, 1)), dtype=self.dtype) * 2
        )
        self.res_ = Tensor(
            np.array(
                np.zeros((self.batch_size, self.out_channels, self.out_size,
                          self.out_size)),
                dtype=self.dtype
            )
        )

    def num_inputs(self):
        """Get layer num."""
        return len(self.featmap_strides)

    def log2(self, value):
        """Compute base 2 logarithm."""
        return self.log(value) / self.log(self.twos)

    def build_roi_layers(self, featmap_strides):
        """Build base roi align layer."""
        roi_layers = []
        for s in featmap_strides:
            layer_cls = ROIAlign(self.out_size, self.out_size,
                                 spatial_scale=1 / s,
                                 sample_num=self.sample_num)
            roi_layers.append(layer_cls)
        return roi_layers

    def _c_map_roi_levels(self, rois: ms.Tensor) -> ms.Tensor:
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = (
            self.sqrt(rois[::, 3:4:1] - rois[::, 1:2:1] + self.ones) *
            self.sqrt(rois[::, 4:5:1] - rois[::, 2:3:1] + self.ones)
        )

        target_lvls = self.log2(scale / self.finest_scale + self.epslion)
        target_lvls = ops.floor(target_lvls)
        target_lvls = self.cast(target_lvls, ms.int32)
        target_lvls = self.clamp(target_lvls, self.zeros, self.max_levels)

        return target_lvls

    def construct(self, rois: ms.Tensor, feats: ms.Tensor) -> ms.Tensor:
        """Extract features."""
        res = self.res_
        target_lvls = self._c_map_roi_levels(rois)
        for i in range(self.num_levels):
            mask = self.equal(target_lvls, ops.scalar_to_tensor(i, ms.int32))
            mask = ops.reshape(mask, (-1, 1, 1, 1))
            roi_feats_t = self.roi_layers[i](feats[i], rois)
            mask = self.cast(
                ops.tile(
                    self.cast(mask, ms.int32),
                    (1, self.out_channels, self.out_size, self.out_size)
                ),
                ms.bool_
            )
            res = self.select(mask, roi_feats_t, res)

        return res
