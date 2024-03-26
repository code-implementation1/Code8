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
"""FCNMaskHead for Mask-RCNN detection models."""
from typing import Dict, Tuple, Union

import cv2
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import ops
import pycocotools.mask as mask_util

from ..layers import ConvModule, build_upsample_layer

from .. import Config


def _pair(x: Union[int, Tuple[int, int]]):
    return (x, x) if isinstance(x, int) else x


def prepare_config(cfg=None):
    if isinstance(cfg, tuple):
        cfg = dict(cfg)
    elif isinstance(cfg, Config):
        cfg = cfg.as_dict().copy()

    return cfg


class FCNMaskHead(nn.Cell):
    """FCNMaskHead for Mask-RCNN detection models.

        Args:
            loss_mask (nn.Cell): Segmentation mask loss.
            train_batch_size (int): Training batch size.
            test_batch_size (int): Test batch size.
            loss_mask_weight (float): Weight of segmentation mask loss.
            num_convs (int): Number of convolution layers.
            roi_feat_size (int): Height and width of input ROI features maps.
            in_channels (int): Number of input channels
            conv_kernel_size (int): Convolution kernel size.
            conv_out_channels (int): Number of output channels.
            num_classes (int): Number of classes.
            class_agnostic (bool): If True, single mask will be generated
                else mask will be generated for each class.
            upsample_cfg (Union[Dict, Tuple]): Configuration for upsampling
                layer.
            conv_cfg (Union[Dict, Tuple, None]): Convolution configuration.
            norm_cfg (Union[Dict, Tuple, None]): Normalization configuration.
            train_cfg (Config): Training config.
            test_cfg (Config): Inference config.
    """

    def __init__(
            self,
            loss_mask: nn.Cell,
            train_batch_size: int,
            test_batch_size: int,
            loss_mask_weight: float = 1.0,
            num_convs: int = 4,
            roi_feat_size: int = 14,
            in_channels: int = 256,
            conv_kernel_size: int = 3,
            conv_out_channels: int = 256,
            num_classes: int = 80,
            class_agnostic: bool = False,
            upsample_cfg: Union[Dict, Tuple] = (
                ('type', 'deconv'), ('scale_factor', 2)
            ),
            conv_cfg: Union[Dict, Tuple] = None,
            norm_cfg: Union[Dict, Tuple] = None,
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
            )
    ):
        """Init FCNMaskHead."""
        super(FCNMaskHead, self).__init__()
        self.upsample_cfg = prepare_config(upsample_cfg)
        if self.upsample_cfg['type'] not in [
                None, 'deconv', 'nearest', 'bilinear'
        ]:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear", "carafe"'.format(
                    self.upsample_cfg['type']))
        self.num_convs = num_convs
        # WARN: roi_feat_size is reserved and not used
        self.roi_feat_size = _pair(roi_feat_size)
        self.mask_size = _pair(train_cfg.rcnn.mask_size)
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = self.upsample_cfg.get('type')
        self.scale_factor = self.upsample_cfg.pop('scale_factor')
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = prepare_config(conv_cfg)
        self.norm_cfg = prepare_config(norm_cfg)
        self.max_per_img = test_cfg.rcnn.max_per_img
        self.test_batch_size = test_batch_size
        self.train_batch_size = train_batch_size
        self.loss_mask_weight = loss_mask_weight

        self.convs = nn.CellList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg
                )
            )
        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels
        )
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            self.upsample_cfg.update(
                in_channels=upsample_in_channels,
                out_channels=self.conv_out_channels,
                kernel_size=self.scale_factor,
                stride=self.scale_factor,
                has_bias=True
            )
        self.upsample = build_upsample_layer(self.upsample_cfg)

        out_channels = 1 if self.class_agnostic else self.num_classes + 1
        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)
        self.conv_logits = nn.Conv2d(
            logits_in_channel, out_channels, 1, has_bias=True
        )
        self.relu = nn.ReLU()

        self.loss_mask = loss_mask

        self.bboxes_range = ms.Tensor(
            np.arange(self.max_per_img * test_batch_size).reshape(-1, 1),
            ms.int32
        )

        self.mask_thr_binary = test_cfg.rcnn.mask_thr_binary
        self.on_value = ms.Tensor(1.0, ms.float32)
        self.off_value = ms.Tensor(0.0, ms.float32)

        rmv_first = np.ones((1, self.num_classes + 1))
        rmv_first[:, 0] = np.zeros((1,))
        self.rmv_first_tensor = ms.Tensor(rmv_first, ms.float32)


    def init_weights(self):
        pass

    def construct(self, x):
        """Forward MaskHead."""
        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        return mask_pred

    def get_targets(self, pos_proposals, gt_masks, mask_divider=1):
        """Compute mask target for each positive proposal in the image."""
        n, maxh, maxw = gt_masks.shape
        pos_proposals = pos_proposals / mask_divider
        x1 = ops.clip(pos_proposals[:, [0]], 0, maxw) / maxw
        x2 = ops.clip(pos_proposals[:, [2]], 0, maxw) / maxw
        y1 = ops.clip(pos_proposals[:, [1]], 0, maxh) / maxh
        y2 = ops.clip(pos_proposals[:, [3]], 0, maxh) / maxh
        rois = ops.concat((y1, x1, y2, x2), axis=1)
        inds = ops.arange(0, n, 1, dtype=ms.int32)
        targets = ops.crop_and_resize(
            ops.cast(gt_masks[:, :, :, None], ms.float32), boxes=rois,
            box_indices=inds, crop_size=self.mask_size
        )
        targets = ops.round(targets)
        targets = ops.squeeze(targets)
        return targets

    def loss(self, seg_logits, labels, weights, seg_targets):
        """Loss method."""
        bbox_weights = ops.cast(
            ops.logical_and(ops.gt(labels, 0), weights), ms.int32
        ) * labels
        seg_targets = ops.tile(
            ops.expand_dims(seg_targets, 1), (1, self.num_classes + 1, 1, 1)
        )
        bbox_weights = ops.cast(
            ops.one_hot(
                bbox_weights, self.num_classes + 1,
                self.on_value, self.off_value
            ),
            ms.float32
        )
        bbox_weights = bbox_weights * self.rmv_first_tensor

        # seg_mask_loss
        seg_targets = ops.cast(seg_targets, ms.float32)
        seg_mask_loss = self.loss_mask(seg_logits, seg_targets)
        seg_mask_loss = ops.reduce_mean(seg_mask_loss, (2, 3))
        seg_mask_loss = seg_mask_loss * bbox_weights

        seg_mask_loss = (
            ops.reduce_sum(seg_mask_loss) /
            (ops.reduce_sum(bbox_weights) + 1e-5)
        )
        seg_mask_loss = seg_mask_loss * self.loss_mask_weight
        return seg_mask_loss

    def choose_masks(self, mask_logits, det_labels):
        """Choose computed masks by labels."""
        det_labels = ops.reshape(det_labels, (-1, 1)) + 1
        indices = ops.concat((self.bboxes_range, det_labels), axis=1)
        pred_masks = ops.gather_nd(mask_logits, indices)
        pred_masks = ops.sigmoid(pred_masks)
        pred_masks = ops.reshape(
            pred_masks,
            (
                self.test_batch_size, self.max_per_img,
                self.mask_size[0], self.mask_size[1]
            )
        )
        return pred_masks

    def get_masks(
            self, mask_pred, det_bboxes, ori_shape,
    ):
        """Get segmentation masks from mask_pred and bboxes.
        Args:
            mask_pred (ndarray): shape (n, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multiscale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (ndarray): shape (n, 4/5)
            ori_shape: original image size
        Returns:
            list[list]: encoded masks
        """
        assert isinstance(mask_pred, np.ndarray)
        # when enabling mixed precision training, mask_pred may be float16
        # numpy array
        mask_pred = mask_pred.astype(np.float32)
        segms = []
        bboxes = det_bboxes[:, :4]

        img_h, img_w = ori_shape.astype(np.int32)

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, :].astype(np.int32)
            w = max(bbox[2], 1)
            h = max(bbox[3], 1)
            w = min(w, img_w - bbox[0])
            h = min(h, img_h - bbox[1])
            if w <= 0 or h <= 0:
                print(
                    f'there is invalid proposal bbox, index={i} bbox={bbox} '
                    f'w={w} h={h}'
                )
                w = max(w, 1)
                h = max(h, 1)

            mask_pred_ = mask_pred[i, :, :]

            bbox_mask = cv2.resize(
                mask_pred_, (w, h), interpolation=cv2.INTER_LINEAR
            )
            bbox_mask = (bbox_mask > self.mask_thr_binary).astype(np.uint8)

            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask

            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F')
            )[0]
            segms.append(rle)

        return segms
