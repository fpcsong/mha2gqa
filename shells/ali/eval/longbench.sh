#! /bin/bash

ROOT_PATH=/mnt/data/songxiaohui
export HF_HOME=$ROOT_PATH/cache

cd /mnt/data/songxiaohui/LongBench
python pred.py --model $1