#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distribute_train.sh [device_num] [vgg_ckpt_path] [dataroot]"
echo "For example: bash run_distribute_train.sh 8 ./vgg.ckpt ./ADEChallengeData2016"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e
# ulimit -u unlimited
# export GLOG_v=1
RANK_SIZE=$1
EXEC_PATH=$(pwd)
echo "$EXEC_PATH"
test_dist_8pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_8pcs.json
    export RANK_SIZE=8
}

test_dist_4pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_4pcs.json
    export RANK_SIZE=4
}

test_dist_2pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_2pcs.json
    export RANK_SIZE=2
}

test_dist_${RANK_SIZE}pcs

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd ../

for((i=0;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cd ./device$i
    mkdir src
    cd ../
    cp main_moco.py ./device$i
    cp -rf src/* ./device$i/src
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > train$i.log
    nohup python3.7 -u main_moco.py --distribute 1  $3> train$i.log 2>&1 &
    echo "$i finish"
    cd ../
done
