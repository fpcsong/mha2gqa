#! /bin/bash

ROOT_PATH=/mnt/data/songxiaohui
export HF_HOME=$ROOT_PATH/cache
export NCCL_DEBUG=WARN
BASEMODEL=$1
MODEL_NAME=$2
TOKENIZER_NAME=$3
MXTP=$4
DESC=$5
NUM_GPUS=8
LR=3e-5

BENCHMARK_PATH=./benchmarking/full
TRAINDATA=$BENCHMARK_PATH
EVALDATA=$BENCHMARK_PATH

OUTDIR=$ROOT_PATH/results/bench/sft/$MODEL_NAME/$DESC

mkdir -p $OUTDIR

rm -rf $OUTDIR/*

deepspeed --num_gpus $NUM_GPUS train.py \
    -it \
    -te \
    -t_data $TRAINDATA \
    -v_data $EVALDATA \
    --model_path $BASEMODEL \
    --model_name $MODEL_NAME \
    --tokenizer_name $TOKENIZER_NAME \
    --bf16 \
    -stage 1 \
    -output \
    -output_dir $OUTDIR \
    -max_len 1024 \
    -m_bsz 1 -e_bsz 4 \
    --warmup_steps 32 \
    --eval_steps 20480 \
    --save_steps $MXTP \
    --lr_scheduler cosine \
    --max_steps $MXTP \
    --max_new_tokens 32 \
    --template_name none \
    -lr $LR \
    -bsz 64 \
    --train_files_pattern  '/*/train/*.jsonl' \
    --val_files_pattern '/*/eval/*.jsonl' \
    --deepspeed true \
    --streaming \
    2>&1 | tee $OUTDIR/tee.log
