#! /bin/bash

ROOT_PATH=/mnt/data/songxiaohui
export HF_HOME=$ROOT_PATH/cache

CKPT_PATH=$1
MODEL_NAME=$2
TOKENIZER_NAME=$3
OUTDIR=$CKPT_PATH
BENCHMARK_PATH=./benchmarking/full

TRAINDATA=$BENCHMARK_PATH
EVALDATA=$BENCHMARK_PATH

for BASEMODEL in $CKPT_PATH/*
do
    if [ -f $BASEMODEL/config.json ]; then
        python benchmark.py -te -v_data $EVALDATA \
        --model_path $BASEMODEL \
        --model_name $MODEL_NAME \
        --tokenizer_name $TOKENIZER_NAME \
        --gen_config default \
        --bf16 \
        -output_dir $OUTDIR \
        -m_bsz 2 \
        -e_bsz 2 \
        -max_len 2048 \
        --template_name none \
        --train_files_pattern  '/entitycombine/trainv1/*.jsonl' \
        --val_files_pattern '/entitycombine/evalv1/*.jsonl' \
        -vllm \
        -output \
        2>&1 | tee $OUTDIR/tee.log
    fi
done