#! /bin/bash

ROOT_PATH=/mnt/data/songxiaohui
export HF_HOME=$ROOT_PATH/cache

CKPT_PATH=$1
MODEL_NAME=$2
TOKENIZER_NAME=$3
QUANT=${4:-'none'}
OUTDIR=$CKPT_PATH/
BENCHMARK_PATH=./benchmarking/datasets

mkdir -p $OUTDIR
TRAINDATA=$BENCHMARK_PATH
EVALDATA=$BENCHMARK_PATH
for BASEMODEL in $CKPT_PATH/*
do
    if [ -f $BASEMODEL/config.json ]; then    
        python benchmark.py -te -v_data $EVALDATA \
        --model_path $BASEMODEL \
        --model_name $MODEL_NAME \
        --gen_config default \
        --bf16 \
        -output_dir $OUTDIR \
        -m_bsz 2 \
        -e_bsz 2 \
        -max_len 4096 \
        --template_name none \
        --val_files_pattern '/*/eval/input*.jsonl' \
        --train_files_pattern  '/*/train/input*.jsonl' \
        -output \
        -vllm \
        --max_new_tokens 512 \
        --vllm_quant $QUANT \
        2>&1 | tee $OUTDIR/tee.log
    fi
done
