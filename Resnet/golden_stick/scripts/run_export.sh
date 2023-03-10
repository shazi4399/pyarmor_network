#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
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

CURPATH="$(dirname "$0")"

if [ $# != 4 ]
then
    echo "Usage: bash run_export.sh [PYTHON_PATH] [CONFIG_FILE] [CHECKPOINT_PATH] [MASK_PATH]"
    echo "PYTHON_PATH represents path to directory of 'export.py'."
    exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    real_path=$(realpath -m $PWD/$1)
    echo "${real_path}"
  fi
}

PYTHON_PATH=$(get_real_path $1)
CONFIG_FILE=$(get_real_path $2)
CKPT_FILE=$(get_real_path $3)
MASK_PATH=$(get_real_path $4)

if [ ! -d $PYTHON_PATH ]
then
    echo "error: PYTHON_PATH=$PYTHON_PATH is not a directory"
exit 1
fi

if [ ! -f $CONFIG_FILE ]
then
    echo "error: CONFIG_FILE=$CONFIG_FILE is not a file"
exit 1
fi

if [ ! -f $CKPT_FILE ]
then
    echo "error: CKPT_FILE=$CKPT_FILE is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

if [ -d "UP_export" ];
then
    rm -rf ./UP_export
fi
mkdir ./UP_export
cp ${PYTHON_PATH}/*.py ./UP_export
cp -r ${CURPATH}/../../src ./UP_export
cd ./UP_export || exit
env > env.log
echo "start export for device $DEVICE_ID"
python export.py --config_path=$CONFIG_FILE --checkpoint_file_path=$CKPT_FILE --mask_path=$MASK_PATH &> log &
cd ..
