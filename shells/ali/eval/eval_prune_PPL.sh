#! /bin/bash

ROOT_PATH=/mnt/data/songxiaohui
export HF_HOME=$ROOT_PATH/cache
export NCCL_DEBUG=WARN
MODEL=$1
MODEL_NAME=$2
CONFIG=$3
OUTDIR=$4

BASEMODEL=$ROOT_PATH/models/$MODEL
TEACHER=$ROOT_PATH/teachers/$MODEL_NAME-benchmark
BENCHMARK_PATH=./benchmarking/datasets
TRAINDATA=$BENCHMARK_PATH
EVALDATA=/mnt/data/JinQingyun/llm-workspace-sxh/benchmarking/datasets/mmlu/eval/mmlu.jsonl
echo $OUTDIR
for CKPT in $OUTDIR/*
do
    if [ -f $CKPT/config.json ]; then
        mkdir -p $CKPT/pruned
        mkdir -p $CKPT/sft_pruned
        cp $BASEMODEL/*.py $CKPT/
        cp $BASEMODEL/*.py $CKPT/pruned/
        python -m scripts.cvt_prune_weight --model_path $CKPT --model_name $MODEL_NAME --prune_config $CONFIG
        
        python A_MMLUevaluate-5shots.py \
        --model_path $CKPT/pruned \
        --val_files_pattern $EVALDATA
    fi
done