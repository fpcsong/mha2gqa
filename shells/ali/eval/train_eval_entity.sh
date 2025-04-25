#! /bin/bash

ROOT_PATH=/mnt/data/songxiaohui
export HF_HOME=$ROOT_PATH/cache

CKPT_PATH=$1
LR=$2
EPOCH=$3
MODEL_NAME=$4
TOKENIZER_NAME=$5
LORA_R=32
LORA_ALPHA=32
OUTDIR=$CKPT_PATH/saved-$LR-$EPOCH
BENCHMARK_PATH=./benchmarking/full

mkdir -p $OUTDIR
TRAINDATA=$BENCHMARK_PATH
EVALDATA=$BENCHMARK_PATH
for BASEMODEL in $CKPT_PATH/*
do
    if [ -f $BASEMODEL/config.json ]; then
        deepspeed --num_gpus 8 benchmark.py \
        -it \
        -t_data $TRAINDATA \
        -v_data $EVALDATA \
        --model_path $BASEMODEL \
        --model_name $MODEL_NAME \
        --tokenizer_name $TOKENIZER_NAME \
        --gen_config default \
        --bf16 \
        -output_dir $OUTDIR \
        -m_bsz 2 \
        -e_bsz 2 \
        -max_len 1024 \
        --epochs $EPOCH \
        --save_steps 36140 \
        --template_name none \
        -lr $LR \
        -bsz 64 \
        --train_files_pattern  '/entitycombine/trainv1/*.jsonl' \
        --val_files_pattern '/entitycombine/evalv1/*.jsonl' \
        -output \
        --deepspeed true \
        2>&1 | tee $OUTDIR/tee.log
    fi
done

for BASEMODEL in $OUTDIR/*
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
        -max_len 1024 \
        --template_name none \
        --train_files_pattern  '/entitycombine/trainv1/*.jsonl' \
        --val_files_pattern '/entitycombine/evalv1/*.jsonl' \
        -output
    fi
done