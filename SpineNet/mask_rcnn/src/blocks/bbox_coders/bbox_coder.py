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
# and modified: https://github.com/open-mmlab/mmdetection
# ============================================================================
"""Bounding boxes coder-decoder."""
from typing import Optional, Sequence, Tuple

import mindspore as ms
from mindspore import nn
from mindspore import ops


class DeltaXYWHBBoxCoder(nn.Cell):
    """Delta XYWH BBox coder.
    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`_,
    this coder encodes bbox (x1, y1, x2, y2) into delta (dx, dy, dw, dh) and
    decodes delta (dx, dy, dw, dh) back to original bbox (x1, y1, x2, y2).
    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
        clip_border (bool): Whether clip the objects outside the
            border of the image. Defaults to True.
        add_ctr_clamp (bool): Whether to add center clamp, when added, the
            predicted box is clamped is its center is too far away from
            the original anchor's center. Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by YOLOF.
            Default 32.
    """

    def __init__(
            self,
            target_means: Sequence[float] = (0., 0., 0., 0.),
            target_stds: Sequence[float] = (1., 1., 1., 1.),
            clip_border: bool = True,
            add_ctr_clamp: bool = False,
            ctr_clamp: int = 32
    ):
        super(DeltaXYWHBBoxCoder, self).__init__()
        self.means = ms.ops.Tensor(target_means, ms.float32).reshape(1, 4)
        self.stds = ms.ops.Tensor(target_stds, ms.float32).reshape(1, 4)
        self.clip_border = clip_border
        self.add_ctr_clamp = add_ctr_clamp
        self.ctr_clamp = ctr_clamp
        self.eps = ms.Tensor(1e-10, ms.float32)
        self.wp_ratio_clip = ms.ops.Tensor(0.016)

    def encode(self, bboxes: ms.Tensor, gt_bboxes: ms.Tensor) -> ms.Tensor:
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.
        Args:
            bboxes (ms.Tensor): Source boxes, e.g., object proposals.
            gt_bboxes (ms.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.
        Returns:
            ms.Tensor: Box transformation deltas
        """
        assert bboxes.shape[0] == gt_bboxes.shape[0]
        assert bboxes.shape[-1] == 4 and gt_bboxes.shape[-1] == 4
        encoded_bboxes = self.bbox2delta(bboxes, gt_bboxes)
        return encoded_bboxes

    def decode(
            self,
            bboxes: ms.Tensor,
            pred_bboxes: ms.Tensor,
            max_shape: Optional[Sequence[int]] = None,
            wh_ratio_clip: Optional[float] = None
    ) -> ms.Tensor:
        """Apply transformation `pred_bboxes` to `boxes`.
        Args:
            bboxes (ms.Tensor): Basic boxes. Shape (N, 4) or (N, 4)
            pred_bboxes (ms.Tensor): Encoded offsets with respect to
                each roi. Has shape (N, num_classes * 4) or (N, 4) or
                (N, num_classes * 4) or (N, 4). Note N = num_anchors * W * H
                when rois is a grid of anchors.Offset encoding follows [1]_.
            max_shape (Sequence[int] or mindspore.Tensor or Sequence[
                Sequence[int]],optional): Maximum bounds for boxes, specifies
                (H, W, C) or (H, W).
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.
        Returns:
            ms.Tensor: Decoded boxes.
        """
        assert pred_bboxes.shape[0] == bboxes.shape[0]
        if len(pred_bboxes.shape) == 3:
            assert pred_bboxes.shape[1] == bboxes.shape[1]
        if not wh_ratio_clip:
            wh_ratio_clip = self.wp_ratio_clip
        decoded_bboxes = self.delta2bbox(
            bboxes, pred_bboxes, max_shape, wh_ratio_clip=wh_ratio_clip
        )

        return decoded_bboxes

    def bbox2delta(self, proposals, gt):
        assert proposals.shape == gt.shape

        proposals = ops.cast(proposals, ms.float32)
        gt = ops.cast(gt, ms.float32)
        px = (proposals[..., 0] + proposals[..., 2]) * 0.5
        py = (proposals[..., 1] + proposals[..., 3]) * 0.5
        pw = proposals[..., 2] - proposals[..., 0] + 1.0
        ph = proposals[..., 3] - proposals[..., 1] + 1.0

        gx = (gt[..., 0] + gt[..., 2]) * 0.5
        gy = (gt[..., 1] + gt[..., 3]) * 0.5
        gw = gt[..., 2] - gt[..., 0] + 1.0
        gh = gt[..., 3] - gt[..., 1] + 1.0

        dx = (gx - px) / pw
        dy = (gy - py) / ph
        dw = ops.log(gw / pw)
        dh = ops.log(gh / ph)
        deltas = ops.stack([dx, dy, dw, dh], axis=-1)
        deltas = (deltas - self.means) / self.stds

        return deltas

    def delta2bbox(
            self,
            rois,
            deltas,
            max_shape: Tuple[int, int] = None,
            wh_ratio_clip: ms.Tensor = ms.Tensor(0.016)
    ):
        """
        Apply deltas to shift/scale base boxes.

        Typically, the rois are anchor or proposed bounding boxes and the
        deltas are network outputs used to shift/scale those boxes.

        Args:
            rois (Tensor): boxes to be transformed. Has shape (N, 4)
            deltas (Tensor): encoded offsets with respect to each roi.
                Has shape (N, 4). Note N = num_anchors * W * H when rois is a grid
                of anchors. Offset encoding follows [1]_.
            max_shape (tuple[int, int]): maximum bounds for boxes. specifies (H, W)
            wh_ratio_clip (float): maximum aspect ratio for boxes.

        Returns:
            Tensor: boxes with shape (N, 4), where columns represent
                tl_x, tl_y, br_x, br_y.

        References:
            .. [1] https://arxiv.org/abs/1311.2524

        Example:
            >>> rois = ms.Tensor([[ 0.,  0.,  1.,  1.],
            >>>                      [ 0.,  0.,  1.,  1.],
            >>>                      [ 0.,  0.,  1.,  1.],
            >>>                      [ 5.,  5.,  5.,  5.]])
            >>> deltas = ms.Tensor([[  0.,   0.,   0.,   0.],
            >>>                        [  1.,   1.,   1.,   1.],
            >>>                        [  0.,   0.,   2.,  -1.],
            >>>                        [ 0.7, -1.9, -0.5,  0.3]])
            >>> delta2bbox(rois, deltas, max_shape=(32, 32))
            tensor([[0.0000, 0.0000, 1.0000, 1.0000],
                    [0.2817, 0.2817, 4.7183, 4.7183],
                    [0.0000, 0.6321, 7.3891, 0.3679],
                    [5.8967, 2.9251, 5.5033, 3.2749]])
        """
        means = self.means
        stds = self.stds
        denorm_deltas = deltas * stds + means
        dx = denorm_deltas[:, 0::4]
        dy = denorm_deltas[:, 1::4]
        dw = denorm_deltas[:, 2::4]
        dh = denorm_deltas[:, 3::4]
        max_ratio = ops.abs(ops.log(wh_ratio_clip))
        dw = ops.clip_by_value(
            dw, clip_value_min=-max_ratio, clip_value_max=max_ratio
        )
        dh = ops.clip_by_value(
            dh, clip_value_min=-max_ratio, clip_value_max=max_ratio
        )
        # Compute center of each roi
        px = ops.expand_dims((rois[:, 0] + rois[:, 2]) * 0.5, axis=-1)
        py = ops.expand_dims((rois[:, 1] + rois[:, 3]) * 0.5, axis=-1)
        # Compute width/height of each roi
        pw = ops.expand_dims(rois[:, 2] - rois[:, 0] + 1.0, axis=-1)
        ph = ops.expand_dims(rois[:, 3] - rois[:, 1] + 1.0, axis=-1)
        # Use exp(network energy) to enlarge/shrink each roi
        gw = pw * ops.exp(dw)
        gh = ph * ops.exp(dh)
        # Use network energy to shift the center of each roi
        gx = px + pw * dx
        gy = py + ph * dy
        # Convert center-xy/width/height to top-left, bottom-right
        x1 = gx - gw * 0.5 + 0.5
        y1 = gy - gh * 0.5 + 0.5
        x2 = gx + gw * 0.5 - 0.5
        y2 = gy + gh * 0.5 - 0.5
        if max_shape is not None:
            x1 = ops.clip_by_value(
                x1, clip_value_min=0, clip_value_max=max_shape[1] - 1
            )
            y1 = ops.clip_by_value(
                y1, clip_value_min=0, clip_value_max=max_shape[0] - 1
            )
            x2 = ops.clip_by_value(
                x2, clip_value_min=0, clip_value_max=max_shape[1] - 1
            )
            y2 = ops.clip_by_value(
                y2, clip_value_min=0, clip_value_max=max_shape[0] - 1
            )
        bboxes = ops.concat([x1, y1, x2, y2], axis=-1)
        return bboxes
