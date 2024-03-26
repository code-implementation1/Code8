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
from typing import Dict, Optional, Sequence, Tuple
from functools import reduce

import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import ops

from ..layers import ConvModule
from ..initialization import normal_init

from .. import Config


class RetinaSepBNHead(nn.Cell):
    """RetinaHead with separate BN.

    In RetinaHead, conv/norm layers are shared across different FPN levels,
    while in RetinaSepBNHead, conv layers are shared across different FPN
    levels, but BN layers are separated.
    """

    def __init__(
            self,
            feature_shapes: Sequence[Tuple[int, int]],
            num_classes,
            in_channels,
            num_ins,
            loss_cls,
            loss_bbox,
            anchor_generator,
            bbox_coder,
            assigner_sampler,
            feat_channels=256,
            stacked_convs=4,
            conv_cfg=None,
            norm_cfg=None,
            train_cfg=Config({}),
            test_cfg=Config({})
    ):
        super(RetinaSepBNHead, self).__init__()
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.num_ins = num_ins

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels

        self.use_sigmoid_cls = True
        self.sampling = True
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes - 1
        else:
            self.cls_out_channels = num_classes

        if self.cls_out_channels <= 0:
            raise ValueError('num_classes={} is too small'.format(num_classes))

        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox

        # generate anchors
        self.anchor_generator = anchor_generator
        self.feature_shapes = feature_shapes
        self.anchor_list = self.anchor_generator.grid_priors(
            self.feature_shapes
        )
        self.num_anchors = self.anchor_generator.num_base_priors[0]
        self.num_bboxes = sum(
            [
                reduce(lambda x, y: x * y, anchors.shape, 1)
                for anchors in self.anchor_list
            ]
        ) // 4
        self.check_valid = ops.CheckValid()

        self.num_levels = len(feature_shapes)
        self.num_level_anchors = [len(a) for a in self.anchor_list]

        self.bbox_coder = bbox_coder
        self.assigner_sampler = assigner_sampler

        (
            self.cls_convs, self.retina_cls, self.reg_convs, self.retina_reg
        ) = self._init_layers()
        self.relu = nn.ReLU()

        self.eps = ms.Tensor(1e-7, ms.float32)
        self.on_value = ms.Tensor(1.0, ms.float32)
        self.off_value = ms.Tensor(0.0, ms.float32)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self._init_weights()

    def _init_layers(self):
        cls_convs = nn.CellList()
        reg_convs = nn.CellList()
        for _ in range(self.num_ins):
            cls_convs_ = nn.CellList()
            reg_convs_ = nn.CellList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                cls_convs_.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg
                    )
                )
                reg_convs_.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg
                    )
                )
            cls_convs.append(cls_convs_)
            reg_convs.append(reg_convs_)
        for i in range(self.stacked_convs):
            for j in range(1, self.num_ins):
                cls_convs[j][i].conv.weight = cls_convs[0][i].conv.weight
                reg_convs[j][i].conv.weight = reg_convs[0][i].conv.weight
        retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            pad_mode='pad',
            padding=1,
            has_bias=True
        )
        retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3,
            pad_mode='pad', padding=1, has_bias=True
        )

        return cls_convs, retina_cls, reg_convs, retina_reg

    def _init_weights(self):
        for m in self.cls_convs[0]:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs[0]:
            normal_init(m.conv, std=0.01)
        bias_cls = float(-np.log((1 - 0.01) / 0.01))
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def construct(self, feats):
        cls_scores = []
        bbox_preds = []
        for i, _ in enumerate(feats):
            cls_feat = feats[i]
            reg_feat = feats[i]
            for cls_conv in self.cls_convs[i]:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs[i]:
                reg_feat = reg_conv(reg_feat)
            cls_score = self.retina_cls(cls_feat)
            bbox_pred = self.retina_reg(reg_feat)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)

        return cls_scores, bbox_preds

    def loss(
            self,
            cls_scores,
            bbox_preds,
            gt_valids,
            gt_bboxes,
            gt_labels,
            img_metas
    ):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            gt_valids (list([Tensor]): Mask that show gt bboxes.
            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # batch_size, num_gts_per_batch = gt_valids.shape[:2]
        # List of anchors for each level of FPN (without BS dim)
        anchor_list = self.anchor_list
        # List of valid flags for ech level of FPN
        # (*currently all True)

        valid_flag_list = [
            ops.stop_gradient(
                ops.check_valid(
                    anchors, ops.concat(
                        (img_metas[0][4:6], ops.ones((1,)))
                    )
                )
            )
            for anchors in anchor_list
        ]

        targets = self.get_targets(
            anchor_list=anchor_list,
            valid_flag_list=valid_flag_list,
            gt_bboxes_list=gt_bboxes,
            img_metas=img_metas,
            gt_valids=gt_valids,
            gt_labels_list=gt_labels
        )
        (
            bbox_targets_list, bbox_weights_list,
            labels_list, label_weights_list, num_total_bboxes
        ) = targets

        losses_cls, losses_bbox = [], []
        # Iterate pyramid levels
        for i in range(len(cls_scores)):
            # cls_scores[i] has shape (B, NUM_CLASSES, H, W)
            # cls_scores[i] has shape (B, 4, H, W)
            # has shape (B, N[i], 4)
            # labels_list[i]: (B, N[i])
            # label_weights_list[i]: (B, N[i], 4)
            # bbox_targets_list[i]: (B, N[i], 4)
            # bbox_weights_list[i]: (B, N[i], 4)
            # where B - batch size, N[i] - number of anchors of i-th FPN level
            loss_cls, loss_bbox = self.loss_single(
                cls_score=cls_scores[i],
                bbox_pred=bbox_preds[i],
                labels=labels_list[i],
                label_weights=label_weights_list[i],
                bbox_targets=bbox_targets_list[i],
                bbox_weights=bbox_weights_list[i],
                num_total_samples=num_total_bboxes
            )
            losses_cls.append(loss_cls)
            losses_bbox.append(loss_bbox)

        return losses_cls, losses_bbox

    def loss_single(
            self,
            cls_score,
            bbox_pred,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            num_total_samples
    ):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # classification loss
        labels = ops.reshape(labels, (-1,))
        labels = ops.one_hot(
            labels - 1, depth=self.cls_out_channels, on_value=self.on_value,
            off_value=self.off_value
        )
        label_weights = ops.reshape(label_weights, (-1, self.cls_out_channels))
        label_weights = ops.cast(label_weights, ms.float32)
        cls_score = ops.reshape(
            ops.transpose(cls_score, (0, 2, 3, 1)),
            (-1, self.cls_out_channels)
        )
        loss_cls = self.loss_cls(cls_score, labels)
        loss_cls = (
            ops.reduce_sum(label_weights * loss_cls) /
            (num_total_samples + self.eps)
        )

        # regression loss
        bbox_targets = ops.reshape(bbox_targets, (-1, 4))
        bbox_weights = ops.reshape(bbox_weights, (-1, 4))
        bbox_weights = ops.cast(bbox_weights, ms.float32)
        bbox_pred = ops.reshape(
            ops.transpose(bbox_pred, (0, 2, 3, 1)), (-1, 4)
        )
        loss_bbox = self.loss_bbox(bbox_pred, bbox_targets)
        loss_bbox = (
            ops.reduce_sum(bbox_weights * loss_bbox) /
            (num_total_samples + self.eps)
        )

        return loss_cls, loss_bbox

    def get_targets(
            self,
            anchor_list,
            valid_flag_list,
            gt_bboxes_list,
            img_metas,
            gt_valids,
            gt_labels_list=None,
    ):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_valids ():
            gt_labels_list (list[Tensor]): Ground truth labels of each box.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each
                  level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all
                  images.
                - num_total_neg (int): Number of negative samples in all
                  images.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        # Number of anchors on every level
        num_level_anchors = self.num_level_anchors
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []

        # For each image in batch concat all anchors
        for i in range(num_imgs):
            concat_anchor_list.append(ops.concat(anchor_list))
            concat_valid_flag_list.append(ops.concat(valid_flag_list))

        # compute targets for each image

        all_labels = []
        all_label_weights = []
        all_bbox_targets = []
        all_bbox_weights = []
        num_total_bboxes = 0.
        # Iterate batch size
        for i in range(num_imgs):
            result = self.assigner_sampler(
                gt_bboxes_i=gt_bboxes_list[i],
                gt_labels_i=gt_labels_list[i],
                bboxes=concat_anchor_list[i],
                gt_valids=gt_valids[i],
                valid_mask=concat_valid_flag_list[i]
            )
            (
                all_bbox_target, all_bbox_weight, all_label, all_label_weight
            ) = result
            all_label_weight = ops.tile(
                ops.expand_dims(all_label_weight, axis=-1),
                (1, self.cls_out_channels)
            )
            all_bbox_weight = ops.tile(
                ops.expand_dims(all_bbox_weight, axis=-1),
                (1, 4)
            )

            all_labels.append(all_label)
            all_label_weights.append(all_label_weight)
            all_bbox_targets.append(all_bbox_target)
            all_bbox_weights.append(all_bbox_weight)

            num_total_bboxes = num_total_bboxes + ops.reduce_sum(
                ops.cast(all_label > 0, ms.float32)
            )

        # split targets to a list w.r.t. multiple levels
        num_total_bboxes = ops.maximum(num_total_bboxes, 1)

        labels_list = images_to_levels(all_labels, num_level_anchors)
        labels_list = [ops.stop_gradient(labels) for labels in labels_list]

        label_weights_list = images_to_levels(
            all_label_weights, num_level_anchors
        )
        label_weights_list = [
            ops.stop_gradient(label_weights)
            for label_weights in label_weights_list
        ]

        bbox_targets_list = images_to_levels(
            all_bbox_targets, num_level_anchors
        )
        bbox_targets_list = [
            ops.stop_gradient(bbox_targets)
            for bbox_targets in bbox_targets_list
        ]

        bbox_weights_list = images_to_levels(
            all_bbox_weights, num_level_anchors
        )
        bbox_weights_list = [
            ops.stop_gradient(bbox_weights)
            for bbox_weights in bbox_weights_list
        ]
        res = (
            bbox_targets_list, bbox_weights_list, labels_list,
            label_weights_list, num_total_bboxes
        )

        return res

    def get_bboxes(
            self,
            bbox_preds,
            cls_scores,
            img_metas=None,
            score_factors=None,
            cfg=None,
            rescale=False,
            with_nms=True,
            **kwargs
    ):
        """Transform network outputs of a batch into bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Default None.
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = [i.asnumpy() for i in
                       self.anchor_generator.grid_priors(featmap_sizes)]

        res_bboxes = []
        res_labels = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            if with_score_factors:
                score_factor_list = select_single_mlvl(score_factors, img_id)
            else:
                score_factor_list = [None for _ in range(num_levels)]
            det_bboxes, det_labels = self._get_bboxes_single(
                cls_score_list,
                bbox_pred_list,
                score_factor_list,
                mlvl_priors,
                img_meta,
                cfg,
                rescale,
                with_nms,
                **kwargs
            )
            res_bboxes.append(det_bboxes)
            res_labels.append(det_labels)

        return np.array(res_bboxes), np.array(res_labels)

    def _get_bboxes_single(
            self,
            cls_score_list,
            bbox_pred_list,
            score_factor_list,
            mlvl_priors,
            img_meta,
            cfg,
            rescale=False,
            with_nms=True,
            **kwargs
    ):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta[2:4]
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for _, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, mlvl_priors)):
            assert cls_score.shape[-2:] == bbox_pred.shape[-2:]
            bbox_pred = np.transpose(bbox_pred, (1, 2, 0)).reshape(-1, 4)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()

            scores = np.transpose(cls_score, (1, 2, 0)).reshape(
                (-1, self.cls_out_channels)
            )
            if self.use_sigmoid_cls:
                scores = 1 / (1 + np.exp(-scores))
            else:
                # remind that we set FG labels to [0, num_class-1]
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.

            idxs, scores, labels = filter_scores_and_topk(
                scores=scores,
                score_thr=cfg.score_thr,
                topk=nms_pre
            )

            bbox_pred = bbox_pred[idxs]
            priors = priors[idxs]

            if with_score_factors:
                score_factor = score_factor[idxs]

            if priors.size != 0 and bbox_pred.size != 0:
                bboxes = self.bbox_coder.decode(
                    ms.Tensor(priors), ms.Tensor(bbox_pred),
                    max_shape=ms.Tensor(img_shape)
                ).asnumpy()
            else:
                bboxes = np.zeros((0, 4))

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        return self._bbox_post_process(
            mlvl_scores, mlvl_labels, mlvl_bboxes, False, cfg, rescale,
            with_nms, mlvl_score_factors, **kwargs
        )

    def _bbox_post_process(
            self,
            mlvl_scores,
            mlvl_labels,
            mlvl_bboxes,
            scale_factor,
            cfg,
            rescale=False,
            with_nms=True,
            mlvl_score_factors=None,
    ):
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            mlvl_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_labels (list[Tensor]): Box class labels from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale
                levels of a single image, each item has shape (num_bboxes, 4).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
            mlvl_score_factors (list[Tensor], optional): Score factor from
                all scale levels of a single image, each item has shape
                (num_bboxes, ). Default: None.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
        assert len(mlvl_scores) == len(mlvl_bboxes) == len(mlvl_labels)
        mlvl_bboxes = np.concatenate(mlvl_bboxes)
        mlvl_scores = np.concatenate(mlvl_scores)
        mlvl_labels = np.concatenate(mlvl_labels)

        if mlvl_score_factors is not None:
            mlvl_score_factors = np.concatenate(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors

        if with_nms:
            if mlvl_bboxes.size == 0:
                det_bboxes = np.concatenate(
                    [mlvl_bboxes, mlvl_scores[:, None]], -1)
                return det_bboxes, mlvl_labels

            det_bboxes, keep_idxs = batched_nms(boxes=mlvl_bboxes,
                                                scores=mlvl_scores,
                                                idxs=mlvl_labels,
                                                nms_cfg=cfg.nms)

            det_bboxes = det_bboxes[:cfg.max_per_img]
            det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img]

            temp_bboxes = [[] for i in range(80)]
            for i, bbox in zip(det_labels, det_bboxes):
                temp_bboxes[i].append(bbox)
            return det_bboxes, det_labels

        return mlvl_bboxes, mlvl_scores, mlvl_labels


def batched_nms(
        boxes: np.ndarray,
        scores: np.array,
        idxs: np.array,
        nms_cfg: Optional[Dict],
        class_agnostic: bool = False
    ) -> Tuple[np.array, np.array]:
    r"""Performs non-maximum suppression in a batched fashion.

    Modified from `torchvision/ops/boxes.py#L39
    <https://github.com/pytorch/vision/blob/
    505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39>`_.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.

    Note:
        In v1.4.1 and later, ``batched_nms`` supports skipping the NMS and
        returns sorted raw results when `nms_cfg` is None.

    Args:
        boxes (np.array): boxes in shape (N, 4) or (N, 5).
        scores (np.array): scores in shape (N, ).
        idxs (np.array): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict | optional): Supports skipping the nms when `nms_cfg`
            is None, otherwise it should specify nms type and other
            parameters like `iou_thr`. Possible keys includes the following.

            - iou_threshold (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
              number of boxes is large (e.g., 200k). To avoid OOM during
              training, the users could set `split_thr` to a small value.
              If the number of boxes is greater than the threshold, it will
              perform NMS on each group of boxes separately and sequentially.
              Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class. Defaults to False.

    Returns:
        tuple: kept dets and indice.

        - boxes (np.array): Bboxes with score after nms, has shape
          (num_bboxes, 5). last dimension 5 arrange as
          (x1, y1, x2, y2, score)
        - keep (np.array): The indices of remaining boxes in input
          boxes.
    """
    # skip nms when nms_cfg is None
    if nms_cfg is None:
        scores, inds = np.sort(scores)[::-1]
        boxes = boxes[inds]
        return np.concatenate([boxes, scores[:, None]], -1), inds

    class_agnostic = nms_cfg.pop('class_agnostic', class_agnostic)

    if class_agnostic:
        boxes_for_nms = boxes
    else:
        # When using rotated boxes, only apply offsets on center.
        if boxes.shape[-1] == 5:
            # Strictly, the maximum coordinates of the rotating box
            # (x,y,w,h,a) should be calculated by polygon coordinates.
            # But the conversion from rotated box to polygon will
            # slow down the speed.
            # So we use max(x,y) + max(w,h) as max coordinate
            # which is larger than polygon max coordinate
            # max(x1, y1, x2, y2,x3, y3, x4, y4)
            max_coordinate = boxes[..., :2].max() + boxes[..., 2:4].max()
            offsets = idxs * (max_coordinate + 1)
            boxes_ctr_for_nms = boxes[..., :2] + offsets[:, None]
            boxes_for_nms = np.concatenate(
                [boxes_ctr_for_nms, boxes[..., 2:5]],
                axis=-1
            )
        else:
            max_coordinate = boxes.max()
            offsets = idxs * (max_coordinate + 1)
            boxes_for_nms = boxes + offsets[:, None]

    max_num = nms_cfg.get('max_num', -1)
    split_thr = nms_cfg.pop('split_thr', 10000)
    if boxes_for_nms.shape[0] < split_thr:
        keep = apply_nms(boxes=boxes_for_nms,
                         scores=scores,
                         thres=nms_cfg.iou_threshold,
                         max_boxes=max_num)
        boxes = boxes[keep]
        scores = scores[keep]
    else:
        total_mask = np.zeros_like(scores, dtype=np.bool)
        # Some type of nms would reweight the score, such as SoftNMS
        scores_after_nms = np.zeros_like(scores)
        for id_ in np.unique(idxs):
            mask = (idxs == id_).nonzero(as_tuple=False).view(-1)
            dets, keep = apply_nms(
                boxes=boxes_for_nms[mask], scores=scores[mask],
                thres=nms_cfg.iou_threshold, max_boxes=max_num
            )
            total_mask[mask[keep]] = True
            scores_after_nms[mask[keep]] = dets[:, -1]
        keep = total_mask.nonzero().view(-1)

        scores, inds = scores_after_nms[keep].sort(descending=True)
        keep = keep[inds]
        boxes = boxes[keep]

        if max_num > 0:
            keep = keep[:max_num]
            boxes = boxes[:max_num]
            scores = scores[:max_num]

    boxes = np.concatenate([boxes, scores[:, None]], -1)
    return boxes, keep


def apply_nms(boxes, scores, thres, max_boxes):
    """Apply NMS to bboxes."""
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if len(keep) >= max_boxes != -1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thres)[0]

        order = order[inds + 1]
    return keep

def filter_scores_and_topk(scores, score_thr, topk):
    """Filter results using score threshold and topk candidates.

    Args:
        scores (np.array): The scores, shape (num_bboxes, K).
        score_thr (float): The score filter threshold.
        topk (int): The number of topk candidates.

    Returns:
        tuple: Filtered results

            - scores (np.array): The scores after being filtered, \
                shape (num_bboxes_filtered, ).
            - labels (np.array): The class labels, shape \
                (num_bboxes_filtered, ).
            - anchor_idxs (np.array): The anchor indexes, shape \
                (num_bboxes_filtered, ).
            - filtered_results (dict or list or np.array, Optional): \
                The filtered results. The shape of each item is \
                (num_bboxes_filtered, N).
    """
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = np.nonzero(valid_mask)

    num_topk = min(topk, len(valid_idxs[0]))
    idxs = np.argsort(scores)[::-1][:num_topk]

    keep_idxs = valid_idxs[0][idxs]
    labals = valid_idxs[1][idxs]

    return keep_idxs, scores[idxs], labals

def select_single_mlvl(mlvl_tensors, batch_id):
    """Extract a multi-scale single image tensor from a multi-scale batch
    tensor based on batch index.

    Note: The default value of detach is True, because the proposal gradient
    needs to be detached during the training of the two-stage model. E.g
    Cascade Mask R-CNN.

    Args:
        mlvl_tensors (list[np.array]): Batch tensor for all scale levels,
           each is a 4D-tensor.
        batch_id (int): Batch index.

    Returns:
        list[np.array]: Multi-scale single image tensor.
    """
    assert isinstance(mlvl_tensors, (list, tuple))
    num_levels = len(mlvl_tensors)

    mlvl_tensor_list = [
        mlvl_tensors[i][batch_id] for i in range(num_levels)
    ]
    return mlvl_tensor_list

def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = ops.stack(target, axis=0)
    level_targets = []
    start = ms.Tensor(0).astype(ms.int32)
    one = ms.Tensor(1., ms.int32)
    for n in num_levels:
        end = start + n
        level_targets.append(
            ops.gather(target, ops.range(start, end, one), axis=1)
        )
        start = end
    return level_targets
