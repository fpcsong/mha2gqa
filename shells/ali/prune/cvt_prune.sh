#! /bin/bash

CKPT_PATH=$1
MODEL_NAME=$2
CONFIG=$3
for BASEMODEL in $CKPT_PATH/*
do
    if [ -f $BASEMODEL/config.json ]; then
        python -m scripts.cvt_prune_weight --model_path $BASEMODEL --prune_config $CONFIG --model_name $MODEL_NAME --attn_impl eager
    fi
done