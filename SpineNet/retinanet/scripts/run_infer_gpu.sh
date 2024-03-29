#!/bin/bash
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

if [ $# != 4 ]
then 
    echo "Usage: 
    bash scripts/run_infer_gpu.sh [CONFIG_PATH] [CHECKPOINT_PATH] [PRED_INPUT] [PRED_OUTPUT]
    "
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

CONFIG_PATH=$(get_real_path $1)
CHECKPOINT_PATH=$(get_real_path $2)
PRED_INPUT=$(get_real_path $3)
PRED_OUTPUT=$(get_real_path $4)


echo "config_path: $CONFIG_PATH"
echo "checkpoint_path: $CHECKPOINT_PATH"
echo "pred_input: $PRED_INPUT"
echo "pred_output: $PRED_OUTPUT"

python infer.py --config_path $CONFIG_PATH --checkpoint_path $CHECKPOINT_PATH --pred_input $PRED_INPUT --pred_output $PRED_OUTPUT $EXTRA 2>&1 | tee infer.log

