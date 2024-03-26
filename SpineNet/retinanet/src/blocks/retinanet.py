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
"""RetinaNet"""
import numpy as np
import mindspore as ms
from mindspore import ops, nn
from mindspore.common.tensor import Tensor

from .backbones import SpineNet
from .bbox_coders import DeltaXYWHBBoxCoder
from .dense_heads import RetinaSepBNHead
from .losses import SigmoidFocalClassificationLoss
from .anchor_generator import AnchorGenerator
from .assigners_samplers import build_mask_max_iou_pseudo


class RetinaNet(nn.Cell):
    """
    RetinaNet Network.

    Examples:
        net = RetinaNet(config)
    """

    def __init__(self, config):
        super().__init__()
        self.dtype = np.float32
        self.ms_type = ms.float32
        self.train_batch_size = config.batch_size
        self.test_batch_size = config.test_batch_size
        self.num_classes = config.num_classes

        # Anchor generator
        self.gt_labels_stage1 = Tensor(
            np.ones((self.train_batch_size, config.num_gts)).astype(np.uint8)
        )

        # Backbone
        self.backbone = self.create_backbone(config)

        loss_cls = SigmoidFocalClassificationLoss(
            gamma=config.bbox_head.loss_cls.gamma,
            alpha=config.bbox_head.loss_cls.alpha
        )
        loss_bbox = ops.SmoothL1Loss(
            beta=config.bbox_head.loss_bbox.beta
        )
        anchor_generator = AnchorGenerator(
            **config.bbox_head.anchor_generator.as_dict()
        )
        bbox_coder = DeltaXYWHBBoxCoder(
            **config.bbox_head.bbox_coder.as_dict()
        )

        num_bboxes = sum(
            anchor_generator.num_base_anchors[i] * sh[0] * sh[1]
            for i, sh in enumerate(config.feature_shapes)
        )

        assigner_sampler = build_mask_max_iou_pseudo(
            num_bboxes=num_bboxes, num_gts=config.num_gts,
            assigner_cfg=config.train_cfg.assigner.as_dict(),
            bbox_coder=bbox_coder
        )

        self.bbox_head = RetinaSepBNHead(
            feature_shapes=config.feature_shapes,
            num_classes=config.num_classes,
            in_channels=config.bbox_head.in_channels,
            num_ins=config.bbox_head.num_ins,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            anchor_generator=anchor_generator,
            bbox_coder=bbox_coder,
            assigner_sampler=assigner_sampler,
            feat_channels=config.bbox_head.feat_channels,
            stacked_convs=config.bbox_head.stacked_convs,
            conv_cfg=None,
            norm_cfg=config.bbox_head.norm_cfg.as_dict(),
            train_cfg=config.train_cfg,
            test_cfg=config.test_cfg
        )

    def create_backbone(self, config):
        """Create backbone and init it."""
        backbone_cfg = config.backbone.as_dict().copy()
        backbone_type = backbone_cfg.pop('type')
        if backbone_type == 'spinenet':
            backbone = SpineNet(
                **backbone_cfg, img_size=(config.img_height, config.img_width)
            )
        else:
            raise ValueError(f'Unsupported backbone: {config.backbone.type}')
        return backbone

    def construct(self, img_data, img_metas, gt_bboxes, gt_labels, gt_valids):
        """
        construct the RetinaNet.

        Args:
            img_data: input image data.
            img_metas: meta label of img.
            gt_bboxes (Tensor): get the value of bboxes.
            gt_labels (Tensor): get the value of labels.
            gt_valids (Tensor): get the valid part of bboxes.

        Returns:
            Tuple,tuple of output tensor (losses if training else predictions).
        """
        x = self.backbone(img_data)

        cls_score, bbox_pred = self.bbox_head(x)

        # gt_bboxes is a list of B ms.Tensors, each has shape (100, 4)
        if self.training:
            losses_cls, losses_bbox = self.bbox_head.loss(
                cls_scores=cls_score, bbox_preds=bbox_pred,
                gt_valids=gt_valids, gt_bboxes=gt_bboxes,
                gt_labels=gt_labels, img_metas=img_metas)
            output = losses_cls, losses_bbox
        else:
            output = cls_score, bbox_pred
        return output

    def set_train(self, mode=True):
        """Change training mode."""
        super().set_train(mode=mode)
        self.backbone.set_train(mode=mode)


class RetinaNetInfer:
    """RetinaNet wrapper for inference."""

    def __init__(self, config):
        super().__init__()
        self.net = RetinaNet(config)
        self.net.set_train(False)

    def __call__(self, img_data, img_metas=None):
        """Make predictions."""
        cls_scores, bbox_preds = self.net(img_data, img_metas,
                                          None, None, None)

        cls_scores = [i.asnumpy() for i in cls_scores]
        bbox_preds = [i.asnumpy() for i in bbox_preds]
        img_metas = [i.asnumpy() for i in img_metas]

        boxes, labels = self.net.bbox_head.get_bboxes(
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            img_metas=img_metas
        )

        return boxes, labels

    def set_train(self, mode):
        self.net.set_train(mode)
