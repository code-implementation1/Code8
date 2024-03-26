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
"""Bounding box head."""
from typing import Sequence, Tuple

import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import ops

from ..layers import ConvModule
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

class ConvFCBBoxHead(nn.Cell):
    """Bounding box head.

    Args:
        train_batch_size (int): Train batch size
        test_batch_size (int): Test batch size
        num_gts (int): count Ground Truth.
        bbox_coder (nn.Cell): Bounding box coder
        loss_cls (nn.Cell): Classification loss block
        loss_bbox (nn.Cell): Regression loss block
        targets_generator (nn.Cell): Block that generates targets. Contain
            assigner, sampler and bbox_coder (encode bounding boxes).
        num_classes (int): number of classification categories
            (excluding background).
        in_channels (int): number of input channels
        fc_out_channels (int): number of output channels of layer before
            classification.
        roi_feat_size (int): h and w size of input feature maps.
        loss_cls_weight (float): Classification loss weight
        loss_bbox_weight (float): Regression loss weight
        train_cfg (Config): Model training configurations
        test_cfg (Config): Model inference configurations
    """

    def __init__(
            self,
            train_batch_size: int,
            test_batch_size: int,
            num_gts: int,
            bbox_coder: nn.Cell,
            loss_cls: nn.Cell,
            loss_bbox: nn.Cell,
            targets_generator: nn.Cell,
            num_classes: int = 80,
            in_channels: int = 256,
            num_shared_convs=0,
            num_shared_fcs=0,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            with_avg_pool=False,
            with_cls=True,
            with_reg=True,
            conv_out_channels=256,
            fc_out_channels: int = 1024,
            roi_feat_size: int = 7,
            reg_class_agnostic=False,
            loss_cls_weight: float = 1.0,
            loss_bbox_weight: float = 1.0,
            train_cfg: Config = Config(
                dict(
                    rpn_proposal=dict(max_per_img=1000)
                )
            ),
            test_cfg: Config = Config(
                dict(
                    rcnn=dict(
                        score_thr=0.05,
                        iou_threshold=0.5,
                        max_per_img=100
                    )
                )
            ),
            conv_cfg=None,
            norm_cfg=None
    ):
        """Init ConvFCBBoxHead."""
        super().__init__()
        conv_cfg, norm_cfg, _ = prepare_layers_configs(
            conv_cfg=conv_cfg, norm_cfg=norm_cfg
        )

        self.ms_type = ms.float32
        self.dtype = np.float32
        self.num_gts = num_gts
        self.in_channels = in_channels
        self.fc_out_channels = fc_out_channels
        self.roi_feat_size = roi_feat_size
        self.roi_feat_area = self.roi_feat_size ** 2
        self.num_classes = num_classes
        self.num_cls_bbox = num_classes + 1
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.reg_class_agnostic = reg_class_agnostic

        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)

        self.loss_cls_weight = ms.Tensor(
            np.array(loss_cls_weight).astype(self.dtype)
        )
        self.loss_reg_weight = ms.Tensor(
            np.array(loss_bbox_weight).astype(self.dtype)
        )

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        # reconstruct fc_cls and fc_reg since input channels are changed
        self.fc_cls = nn.Dense(last_layer_dim, self.num_cls_bbox)

        # reconstruct fc_cls and fc_reg since input channels are changed
        self.fc_reg = nn.Dense(
            last_layer_dim,
            4 if self.reg_class_agnostic else self.num_cls_bbox * 4
        )

        self.relu = ms.ops.ReLU()
        self.flatten = ms.ops.Flatten()
        self.loss_cls = loss_cls

        self.loss_bbox = loss_bbox
        self.bbox_coder = bbox_coder

        self.reshape = ms.ops.Reshape()
        self.onehot = ms.ops.OneHot()
        self.squeeze = ms.ops.Squeeze()
        self.softmax = ms.ops.Softmax(axis=1)
        self.split = ms.ops.Split(axis=0, output_num=self.test_batch_size)
        self.split_shape = ms.ops.Split(axis=0, output_num=4)

        self.on_value = ms.Tensor(1.0, ms.float32)
        self.off_value = ms.Tensor(0.0, ms.float32)
        self.value = ms.Tensor(1.0, self.ms_type)
        self.eps = ms.Tensor(1e-10, self.ms_type)

        self.bbox_coder = bbox_coder

        self.targets_generator = targets_generator

        self.rpn_max_num = test_cfg.rpn.max_per_img
        self.max_num = test_cfg.rcnn.max_per_img
        self.score_thr = test_cfg.rcnn.score_thr
        self.iou_threshold = test_cfg.rcnn.iou_threshold

        self.test_topk = ms.ops.TopK(sorted=True)
        self.nms_test = ms.ops.NMSWithMask(self.iou_threshold)

        ones_mask = np.ones((self.rpn_max_num, 1)).astype(np.bool)
        zeros_mask = np.zeros((self.rpn_max_num, 1)).astype(np.bool)
        self.bbox_mask = ms.Tensor(
            np.concatenate(
                (ones_mask, zeros_mask, ones_mask, zeros_mask),
                axis=1
            )
        )

        self.test_score_thresh = ms.Tensor(
            np.ones((self.rpn_max_num, 1))
            .astype(self.dtype) * self.score_thr
        )
        self.test_score_zeros = ms.Tensor(
            np.ones((self.rpn_max_num, 1))
            .astype(self.dtype) * 0
        )
        self.test_box_zeros = ms.Tensor(
            np.ones((self.rpn_max_num, 4))
            .astype(self.dtype) * -1
        )

        rmv_first = np.ones((1, self.num_classes + 1))
        rmv_first[:, 0] = np.zeros((1,))
        self.rmv_first_tensor = ms.Tensor(rmv_first, ms.float32)

    def _add_conv_fc_branch(
            self,
            num_branch_convs,
            num_branch_fcs,
            in_channels,
            is_shared=False
    ):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = []
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels
                )
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg
                    )
                )
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = []
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels
                )
                branch_fcs.append(
                    nn.Dense(
                        in_channels=fc_in_channels,
                        out_channels=self.fc_out_channels
                    )
                )
            last_layer_dim = self.fc_out_channels

        return (
            nn.CellList(branch_convs), nn.CellList(branch_fcs), last_layer_dim
        )

    def construct(self, x: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor]:
        """Forward BboxHead.

        Args:
            x (ms.Tensor):
                Input features.

        Returns:
            ms.Tensor: classification results.
            ms.Tensor: localization results.
        """
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = ops.flatten(x)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if len(x_cls.shape) > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = ops.flatten(x_cls)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if len(x_reg.shape) > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = ops.flatten(x_reg)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred

    def loss(
            self,
            cls_score: ms.Tensor,
            bbox_pred: ms.Tensor,
            bbox_targets: ms.Tensor,
            labels: ms.Tensor,
            weights: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
        """Loss method.

        Args:
            cls_score (ms.Tensor): Predictions for classification branch.
            bbox_pred (ms.Tensor): Predictions for detection branch.
            bbox_targets (ms.Tensor): Target regressions.
            labels (ms.Tensor): Target classifications.
            weights (ms.Tensor): Mask that show that object is valid.

        Returns:
            ms.Tensor: Total loss
            ms.Tensor: Classification loss
            ms.Tensor: Regression loss
        """
        bbox_weights = self.cast(
            ms.ops.logical_and(ms.ops.gt(labels, 0), weights),
            ms.int32
        ) * labels
        labels = self.onehot(labels, self.num_classes + 1, self.on_value,
                             self.off_value)
        loss_cls = self.loss_cls(cls_score, labels)
        weights = self.cast(weights, self.ms_type)
        loss_cls = loss_cls * weights
        loss_cls = ms.ops.reduce_sum(loss_cls, (0,)) / ms.ops.reduce_sum(
            weights, (0,))

        if self.reg_class_agnostic:
            bbox_weights = self.cast(bbox_weights, self.ms_type)
            bbox_weights = ms.ops.tile(
                ms.ops.expand_dims(bbox_weights, -1), (1, 4)
            )
            pos_bbox_pred = bbox_pred
        else:
            bbox_weights = self.onehot(
                bbox_weights, self.num_classes + 1,
                self.on_value, self.off_value
            )
            bbox_weights = bbox_weights * self.rmv_first_tensor
            bbox_weights = self.cast(bbox_weights, self.ms_type)
            bbox_weights = ms.ops.tile(
                ms.ops.expand_dims(bbox_weights, -1), (1, 1, 4)
            )
            pos_bbox_pred = self.reshape(bbox_pred, (-1, self.num_cls_bbox, 4))
            bbox_targets = ms.ops.expand_dims(bbox_targets, 1)
            bbox_targets = ops.tile(bbox_targets, (1, self.num_cls_bbox, 1))
        loss_reg = self.loss_bbox(pos_bbox_pred, bbox_targets) * bbox_weights
        loss_reg = (
            ms.ops.reduce_sum(loss_reg) /
            (ms.ops.reduce_sum(weights) + self.eps)
        )

        loss = self.loss_cls_weight * loss_cls + self.loss_reg_weight * loss_reg

        return loss, loss_cls, loss_reg

    def get_targets(
            self,
            gt_bboxes: ms.Tensor,
            gt_seg_masks: ms.Tensor,
            gt_labels: ms.Tensor,
            gt_valids: ms.Tensor,
            proposal: Sequence[ms.Tensor],
            proposal_mask: Sequence[ms.Tensor]
    ) -> Tuple[
        Sequence[ms.Tensor], Sequence[ms.Tensor], Sequence[ms.Tensor],
        Sequence[ms.Tensor], Sequence[ms.Tensor], Sequence[ms.Tensor],
        Sequence[ms.Tensor], Sequence[ms.Tensor]
    ]:
        """Prepare proposed samples for training.

        Args:
            gt_bboxes (ms.Tensor): Ground truth bounding bboxes. Shape
                (n, self.num_gts, 4).
            gt_seg_masks (ms.Tensor): Ground truth segmentation masks.
            gt_labels (ms.Tensor): Ground truth labels for training. Shape
                (n, self.num_gts).
            gt_valids (ms.Tensor): Mask that show what GT bboxes are valid.
                Shape (n, self.num_gts).
            proposal (Sequence[ms.Tensor]): Proposed bounding bboxes. Shape
                (n, self.train_cfg.rpn_proposed.max_per_img).
            proposal_mask (Sequence[ms.Tensor]): Mask that shows what proposed
                bounding boxes are valid. Shape
                (n, self.train_cfg.rpn_proposed.max_per_img).

        Returns:
            Sequence[ms.Tensor]: Sampled bounding boxes.
            Sequence[ms.Tensor]: Localization targets for each bbox.
            Sequence[ms.Tensor]: Classification targets for each bbox.
            Sequence[ms.Tensor]: Masks for sampled bounding boxes.
            Sequence[ms.Tensor]: Sampled positive bounding boxes.
            Sequence[ms.Tensor]: Segmentation mask for each positive bbox.
            Sequence[ms.Tensor]: Classification targets for each positive bbox.
            Sequence[ms.Tensor]: Masks for sampled positive bounding boxes.
        """
        bboxes_tuple = []
        targets_tuple = []
        labels_tuple = []
        mask_tuple = []
        pos_bboxes_tuple = []
        seg_tuple = []
        pos_labels_tuple = []
        pos_mask_tuple = []
        for i in range(self.train_batch_size):
            gt_bboxes_i = self.squeeze(gt_bboxes[i:i + 1:1, ::])
            gt_labels_i = self.squeeze(gt_labels[i:i + 1:1, ::])
            gt_labels_i = self.cast(gt_labels_i, ms.uint8)
            gt_valids_i = self.squeeze(gt_valids[i:i + 1:1, ::])
            gt_valids_i = self.cast(gt_valids_i, ms.bool_)
            gt_seg_masks_i = gt_seg_masks[i]
            (
                bboxes, deltas, labels, mask, pos_bboxes, seg, pos_labels,
                pos_mask
            ) = self.targets_generator(
                gt_bboxes_i=gt_bboxes_i, gt_seg_masks_i=gt_seg_masks_i,
                gt_labels_i=gt_labels_i, valid_mask=proposal_mask[i],
                bboxes=proposal[i][::, 0:4:1], gt_valids=gt_valids_i
            )

            bboxes_tuple.append(bboxes)
            labels_tuple.append(labels)
            targets_tuple.append(deltas)
            mask_tuple.append(mask)

            pos_bboxes_tuple.append(pos_bboxes)
            seg_tuple.append(seg)
            pos_labels_tuple.append(pos_labels)
            pos_mask_tuple.append(pos_mask)

        return (
            bboxes_tuple, targets_tuple, labels_tuple, mask_tuple,
            pos_bboxes_tuple, seg_tuple, pos_labels_tuple, pos_mask_tuple
        )

    def get_det_bboxes(
            self,
            cls_logits: ms.Tensor,
            reg_logits: ms.Tensor,
            mask_logits: ms.Tensor,
            rois: ms.Tensor,
            img_metas: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
        """Get the actual detection box.

        Args:
            cls_logits (ms.Tensor): Classification predictions
            reg_logits (ms.Tensor): Localization predictions
            mask_logits (ms.Tensor): Mask that shows actual predictions
            rois (ms.Tensor): Anchors
            img_metas (ms.Tensor): Information about original and final image
            sizes.

        Returns:
            ms.Tensor: Predicted bboxes.
            ms.Tensor: Predicted labels.
            ms.Tensor: Masks that shows actual bboxes.
        """
        scores = self.softmax(cls_logits)

        boxes_all = []
        for i in range(self.num_cls_bbox):
            k = i * 4
            if self.reg_class_agnostic:
                reg_logits_i = self.squeeze(reg_logits)
            else:
                reg_logits_i = self.squeeze(reg_logits[::, k:k + 4:1])
            out_boxes_i = self.bbox_coder.decode(rois, reg_logits_i)
            boxes_all.append(out_boxes_i)

        scores_all = self.split(scores)
        mask_all = self.split(self.cast(mask_logits, ms.int32))

        boxes_all_with_batchsize = []
        for i in range(self.test_batch_size):
            boxes_tuple = []
            for j in range(self.num_cls_bbox):
                boxes_tmp = self.split(boxes_all[j])
                boxes_tuple.append(boxes_tmp[i])
            boxes_all_with_batchsize.append(boxes_tuple)

        res_bboxes_nms, res_labels_nms, res_mask_nms = self.multiclass_nms(
            boxes_all=boxes_all_with_batchsize, scores_all=scores_all,
            mask_all=mask_all
        )

        res_bboxes = []
        res_labels = []
        res_mask = []
        for i in range(self.test_batch_size):
            res_bboxes_, res_labels_, res_mask_ = self.get_best(
                res_bboxes_nms[i], res_labels_nms[i], res_mask_nms[i]
            )
            res_bboxes.append(res_bboxes_)
            res_labels.append(res_labels_)
            res_mask.append(res_mask_)

        res_bboxes = ms.ops.concat(res_bboxes).reshape((-1, self.max_num, 5))
        res_labels = ms.ops.concat(res_labels).reshape((-1, self.max_num, 1))
        res_mask = ms.ops.concat(res_mask).reshape((-1, self.max_num, 1))
        return res_bboxes, res_labels, res_mask

    def get_best(
            self, bboxes: ms.Tensor, labels: ms.Tensor, masks: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
        """Filter predicted bboxes by score."""
        score = bboxes[::, 4] * masks.reshape(-1)
        _, score_indicies = self.test_topk(score, self.max_num)

        bboxes = bboxes[score_indicies]
        labels = labels[score_indicies]
        masks = masks[score_indicies]

        return bboxes, labels, masks

    def multiclass_nms(
            self,
            boxes_all: Sequence[Sequence[ms.Tensor]],
            scores_all: Sequence[ms.Tensor],
            mask_all: Sequence[ms.Tensor]
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
        """Bounding boxes postprocessing."""
        all_bboxes = []
        all_labels = []
        all_masks = []

        for i in range(self.test_batch_size):
            bboxes = boxes_all[i]
            scores = scores_all[i]
            masks = ms.ops.cast(mask_all[i], ms.bool_)

            res_boxes_tuple = []
            res_labels_tuple = []
            res_masks_tuple = []
            for j in range(self.num_classes):
                k = j + 1
                _cls_scores = scores[::, k:k + 1:1]
                _bboxes = ms.ops.squeeze(bboxes[k])
                _mask_o = ms.ops.reshape(masks, (self.rpn_max_num, 1))

                cls_mask = ms.ops.gt(_cls_scores, self.test_score_thresh)
                _mask = ms.ops.logical_and(_mask_o, cls_mask)

                _reg_mask = ms.ops.cast(
                    ms.ops.tile(ms.ops.cast(_mask, ms.int32), (1, 4)), ms.bool_
                )

                _bboxes = ms.ops.select(_reg_mask,
                                        _bboxes,
                                        self.test_box_zeros)
                _cls_scores = ms.ops.select(
                    _mask, _cls_scores, self.test_score_zeros
                )
                __cls_scores = ms.ops.squeeze(_cls_scores)
                scores_sorted, topk_inds = self.test_topk(
                    __cls_scores, self.rpn_max_num
                )
                topk_inds = ms.ops.reshape(topk_inds, (self.rpn_max_num, 1))
                scores_sorted = ms.ops.reshape(
                    scores_sorted, (self.rpn_max_num, 1)
                )
                _bboxes_sorted = ms.ops.gather_nd(_bboxes, topk_inds)
                _mask_sorted = ms.ops.gather_nd(_mask, topk_inds)

                scores_sorted = ms.ops.tile(scores_sorted, (1, 4))
                cls_dets = ms.ops.concat(
                    (_bboxes_sorted, scores_sorted), axis=1
                )
                cls_dets = ms.ops.slice(cls_dets, (0, 0),
                                        (self.rpn_max_num, 5))

                cls_dets, _index, _mask_nms = self.nms_test(cls_dets)
                cls_dets, _index, _mask_nms = [
                    ms.ops.stop_gradient(a)
                    for a in (cls_dets, _index, _mask_nms)
                ]

                _index = ms.ops.reshape(_index, (self.rpn_max_num, 1))
                _mask_nms = ms.ops.reshape(_mask_nms, (self.rpn_max_num, 1))

                _mask_n = ms.ops.gather_nd(_mask_sorted, _index)
                _mask_n = ms.ops.logical_and(_mask_n, _mask_nms)
                cls_labels = ms.ops.ones_like(_index) * j
                res_boxes_tuple.append(cls_dets)
                res_labels_tuple.append(cls_labels)
                res_masks_tuple.append(_mask_n)

            res_boxes = ops.concat(res_boxes_tuple, axis=0)
            res_labels = ops.concat(res_labels_tuple, axis=0)
            res_masks = ops.concat(res_masks_tuple, axis=0)

            reshape_size = self.num_classes * self.rpn_max_num
            res_boxes = ms.ops.reshape(res_boxes, (1, reshape_size, 5))
            res_labels = ms.ops.reshape(res_labels, (1, reshape_size, 1))
            res_masks = ms.ops.reshape(res_masks, (1, reshape_size, 1))

            all_bboxes.append(res_boxes)
            all_labels.append(res_labels)
            all_masks.append(res_masks)

        all_bboxes = ms.ops.concat(all_bboxes, axis=0)
        all_labels = ms.ops.concat(all_labels, axis=0)
        all_masks = ms.ops.concat(all_masks, axis=0)
        return all_bboxes, all_labels, all_masks
