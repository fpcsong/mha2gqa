#! /bin/bash

ROOT_PATH=/mnt/data/songxiaohui
export HF_HOME=$ROOT_PATH/cache
export NCCL_DEBUG=WARN

MODEL_NAME=llama
OUTDIR=$1
# CONFIG=$2 #./prune_utils/config/llama2-1.3B-gqa-8groups.yaml

BASEMODEL=/workspace/Sheared-LLaMA-1.3B
TEACHER=$ROOT_PATH/teachers/$MODEL_NAME-benchmark
BENCHMARK_PATH=./benchmarking/datasets
TRAINDATA=$BENCHMARK_PATH
EVALDATA=$BENCHMARK_PATH

for CKPT in $OUTDIR/*
do
    echo $CKPT
    if [ -f $CKPT/config.json ]; then
        # mkdir -p $CKPT/pruned
        # mkdir -p $CKPT/sft_pruned
        # cp $BASEMODEL/*.py $CKPT/
        # cp $BASEMODEL/*.py $CKPT/pruned/
        # python -m scripts.cvt_prune_weight --model_path $CKPT --model_name $MODEL_NAME --prune_config $CONFIG
        
        python benchmark.py -te -v_data $EVALDATA \
        --model_path $CKPT/pruned \
        --model_name $MODEL_NAME \
        --bf16 \
        -output_dir $OUTDIR \
        -m_bsz 2 \
        -e_bsz 2 \
        -max_len 2048 \
        --template_name none \
        --val_files_pattern '/*/eval/input*.jsonl' \
        --train_files_pattern  '/*/train/input*.jsonl' \
        --eval_vllm \
        -output
    fi
done