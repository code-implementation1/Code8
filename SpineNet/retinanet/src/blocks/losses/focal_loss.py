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
from mindspore import nn, ops
import mindspore as ms


class SigmoidFocalClassificationLoss(nn.Cell):
    """
    Sigmoid focal-loss for classification.

    Args:
        gamma (float): Hyper-parameter to balance the easy and hard examples. Default: 2.0
        alpha (float): Hyper-parameter to balance the positive and negative example. Default: 0.25

    Returns:
        Tensor, the focal loss.
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.sigmiod_cross_entropy = ops.SigmoidCrossEntropyWithLogits()
        self.gamma = gamma
        self.alpha = alpha

    def construct(self, logits, label):
        sigmoid = ops.sigmoid(logits)
        label = ops.cast(label, ms.float32)

        sigmiod_cross_entropy = self.sigmiod_cross_entropy(logits, label)

        p_t = label * sigmoid + (1 - label) * (1 - sigmoid)
        modulating_factor = ops.pow(1 - p_t, self.gamma)

        alpha_weight_factor = label * self.alpha + (1 - label) * (1 - self.alpha)

        focal_loss = modulating_factor * alpha_weight_factor * sigmiod_cross_entropy

        return focal_loss
