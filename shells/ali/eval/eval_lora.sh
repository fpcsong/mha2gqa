#! /bin/bash

ROOT_PATH=/mnt/data/songxiaohui
export HF_HOME=$ROOT_PATH/cache

CKPT_PATH=$1
LR=$2
MODEL_NAME=$3
TOKENIZER_NAME=$4
MBSZ=$5
LOSS_TYPE=${6:-topk_normed_ce}
LORA_R=${7:-8}
LORA_ALPHA=${8:-32}
OUTDIR=$CKPT_PATH/rank-$LORA_R-alpha-$LORA_ALPHA-$LOSS_TYPE
BENCHMARK_PATH=./benchmarking/full

mkdir -p $OUTDIR
TRAINDATA=$BENCHMARK_PATH
EVALDATA=$BENCHMARK_PATH
for BASEMODEL in $CKPT_PATH/*
do
    if [ -f $BASEMODEL/config.json ]; then
        deepspeed --hostfile /etc/mpi/hostfile benchmark.py \
        -it \
        -t_data $TRAINDATA \
        -te \
        -v_data $EVALDATA \
        --model_path $BASEMODEL \
        --model_name $MODEL_NAME \
        --tokenizer_name $TOKENIZER_NAME \
        --gen_config default \
        --bf16 \
        -output_dir $OUTDIR \
        -m_bsz $MBSZ \
        -e_bsz $MBSZ \
        --stage 1 \
        --gradient_checkpointing \
        --temperature 3 \
        -max_len 2048 \
        --epochs 2 \
        --save_steps 36140 \
        --loss_type $LOSS_TYPE \
        --template_name none \
        -lr $LR \
        --lora \
        --dora \
        --lora_r $LORA_R \
        --lora_alpha $LORA_ALPHA \
        -bsz 64 \
        --train_files_pattern  '/*/train/*.jsonl' \
        --val_files_pattern '/*/eval/*.jsonl' \
        -output \
        --deepspeed true \
        2>&1 | tee $OUTDIR/tee.log
    fi
done