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
"""Wrapper for assigner and sampler that generate targets."""
from typing import Tuple

import mindspore as ms
from mindspore import nn
from mindspore import ops

class AssignerPseudoSampler(nn.Cell):
    """Wrapper for pair assigner-sampler.
    Prepares bboxes, calls assigner, add additional gt proposals (if
    necessary), calls sampler, prepares results.

    Args:
        assigner (nn.Cell): Used bounding boxes assigner.
        bbox_coder (nn.Cell): Used bbox coder to generate target values by
            sampled and gt samples.

    Examples:
    """

    def __init__(
            self, assigner: nn.Cell, bbox_coder: nn.Cell
    ):
        """Init AssignerSampler."""
        super().__init__()
        self.assigner = assigner
        self.bbox_coder = bbox_coder

    def prepare(
            self, gt_bboxes_i: ms.Tensor, gt_valids: ms.Tensor,
            bboxes: ms.Tensor, valid_mask: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        """Prepare gt_bboxes_i and bboxes before assignment.

        Args:
            gt_bboxes_i (ms.Tensor): GT bboxes for current image in batch.
            gt_valids (ms.Tensor): Mask that shows which GT bboxes are valid.
            bboxes (ms.Tensor): Proposed bboxes.
            valid_mask (ms.Tensor): Mask that shows which boxes ara valid and
                can be used in assigning and sampling.

        Returns:
            ms.Tensor: Prepared GT bboxes.
            ms.Tensor: Prepared proposed bboxes.
        """
        gt_bboxes_i_mask = ops.cast(
            ops.tile(
                ops.reshape(
                    ops.cast(gt_valids, ms.int32),
                    (self.assigner.num_gts, 1)
                ),
                (1, 4)
            ),
            ms.bool_
        )
        gt_bboxes_i = ops.select(
            gt_bboxes_i_mask, gt_bboxes_i, self.assigner.check_gt_one
        )
        bboxes_mask = ops.cast(
            ops.tile(
                ops.reshape(
                    ops.cast(valid_mask, ms.int32),
                    (self.assigner.num_bboxes, 1)
                ),
                (1, 4)
            ),
            ms.bool_
        )
        bboxes = ops.select(
            bboxes_mask, bboxes, self.assigner.check_anchor_two
        )

        return gt_bboxes_i, bboxes

    def get_result(
            self, assigned_gt_inds: ms.Tensor, bboxes: ms.Tensor,
            gt_bboxes_i: ms.Tensor, assigned_labels: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
        """Prepare result of sampling as targets and weights.

        Args:
            assigned_gt_inds (ms.Tensor): Assigning result.
            bboxes (ms.Tensor): Proposed bboxes.
            assigned_labels (ms.Tensor): GT labels for current image.
            gt_bboxes_i (ms.Tensor): GT bboxes for current image.

        Returns:
             ms.Tensor: Localization targets.
             ms.Tensor: Localization mask.
             ms.Tensor: Classification targets.
             ms.Tensor: Classification mask.
        """
        pos_mask = assigned_gt_inds > 0
        neg_mask = assigned_gt_inds == 0
        assigned_mask = ops.logical_or(pos_mask, neg_mask)

        assigned_gt_inds_ = assigned_gt_inds - 1
        assigned_gt_inds_ = ops.select(
            pos_mask, assigned_gt_inds_,
            ops.zeros_like(assigned_gt_inds)
        )
        assigned_gt_inds_ = ops.reshape(
            assigned_gt_inds_, (-1, 1)
        )
        assigned_gt_bboxes = ops.gather_nd(gt_bboxes_i, assigned_gt_inds_)
        bbox_targets = self.bbox_coder.encode(bboxes, assigned_gt_bboxes)

        bbox_weights = pos_mask
        labels_weights = assigned_mask
        output = bbox_targets, bbox_weights, assigned_labels, labels_weights
        return output

    def construct(
            self, gt_bboxes_i: ms.Tensor, gt_labels_i: ms.Tensor,
            bboxes: ms.Tensor, gt_valids: ms.Tensor, valid_mask: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
        """Assign and sample bboxes.

        Args:
            gt_bboxes_i (ms.Tensor): GT bboxes. Tensor has a fixed shape
                (n, 4), where n equals to self.assigner.num_gts.
            gt_labels_i (ms.Tensor): GT labels. Tensor has a fixed shape (n,),
                where n equals to self.assigner.num_gts.
            bboxes (ms.Tensor): Proposed bboxes or anchors. Tensor has a fixed
                shape (n, 4), where n is self.assigner.num_bboxes.
            gt_valids (ms.Tensor): Mask that shows valid GT bboxes. Tensor has
                a fixed shape (n,), where n equals to self.assigner.num_gts.
            valid_mask (ms.Tensor): Mask that shows valid proposed bboxes.
                Tensor has a fixed shape (n, 4), where n is
                self.assigner.num_bboxes.

        Returns:
            ms.Tensor: Sampled bboxes with shape
                (self.sampler.num_expected_neg + self.sampler.num_expected_pos, 4)
                if self.rcnn_mode else or localization target deltas with shape
                (self.sampler.num_bboxes, 4).
            ms.Tensor: Localization target delta with shape
                (self.sampler.num_expected_neg + self.sampler.num_expected_pos, 4)
                if self.rcnn_mode else localization mask with shape
                (self.sampler.num_bboxes, ) (show what bbox was sampled for
                training).
            ms.Tensor: Classification targets. Tensor has a fixed shape
                (self.sampler.num_expected_neg + self.sampler.num_expected_pos,) if
                self.rcnn_mode else (self.sampler.num_bboxes). Show labels of
                positive bboxes.
            ms.Tensor: Classification mask. Tensor has a fixed shape
                (self.sampler.num_expected_neg + self.sampler.num_expected_pos,) if
                self.rcnn_mode else (self.sampler.num_bboxes). Show what labels
                are valid (maybe used to train classifier).
        """
        gt_bboxes_i, bboxes = self.prepare(
            gt_bboxes_i=gt_bboxes_i, gt_valids=gt_valids, bboxes=bboxes,
            valid_mask=valid_mask
        )
        (
            assigned_gt_inds, _, _, assigned_labels
        ) = self.assigner(
            gt_bboxes_i=gt_bboxes_i, gt_labels_i=gt_labels_i,
            valid_mask=valid_mask, bboxes=bboxes
        )
        return self.get_result(
            assigned_gt_inds, bboxes, gt_bboxes_i,
            assigned_labels
        )
