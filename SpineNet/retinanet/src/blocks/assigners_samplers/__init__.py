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
from .max_iou_assigner import MaxIOUAssigner
from .assigner_pseudo_sampler import AssignerPseudoSampler

from ..bbox_coders import DeltaXYWHBBoxCoder

def build_mask_max_iou_pseudo(
        num_bboxes: int,
        num_gts: int,
        assigner_cfg: dict,
        bbox_coder: DeltaXYWHBBoxCoder,
) -> AssignerPseudoSampler:
    """Build MaxIOUAssigner and PseudoSampler for detection model."""
    assigner = MaxIOUAssigner(
        num_bboxes=num_bboxes, num_gts=num_gts, **assigner_cfg
    )
    assigner_sampler_obj = AssignerPseudoSampler(
        assigner=assigner, bbox_coder=bbox_coder
    )
    return assigner_sampler_obj

__all__ = [
    'MaxIOUAssigner', 'AssignerPseudoSampler', 'build_mask_max_iou_pseudo'
]
