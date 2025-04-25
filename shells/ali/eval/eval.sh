#! /bin/bash

ROOT_PATH=/mnt/data/songxiaohui
export HF_HOME=$ROOT_PATH/cache

CKPT_PATH=$1
LR=$2
MODEL_NAME=$3
TOKENIZER_NAME=$4
LOSS_TYPE=${5:-topk_normed_ce}
MBSZ=${6:-1}
ATTN_IMPL=${7:-flash_attention_2}
OUTDIR=$CKPT_PATH/saved-$LOSS_TYPE
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
        -v_data $EVALDATA \
        --model_path $BASEMODEL \
        --model_name $MODEL_NAME \
        --tokenizer_name $TOKENIZER_NAME \
        --bf16 \
        -output_dir $OUTDIR \
        -m_bsz $MBSZ \
        -e_bsz $MBSZ \
        --stage 1 \
        --gradient_checkpointing \
        -max_len 2048 \
        --epochs 2 \
        --attn_implementation $ATTN_IMPL \
        --temperature 3.0 \
        --save_steps 36140 \
        --loss_type $LOSS_TYPE \
        --template_name none \
        -lr $LR \
        -bsz 64 \
        --train_files_pattern  '/*/train/*.jsonl' \
        --val_files_pattern '/*/eval/*.jsonl' \
        -output \
        --deepspeed true \
        2>&1 | tee $OUTDIR/tee.log
    fi
done

rm -r $OUTDIR/checkpoint-36*

for BASEMODEL in $OUTDIR/*
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
        -max_len 2048 \
        --template_name none \
        --val_files_pattern '/*/eval/*.jsonl' \
        --train_files_pattern  '/*/train/*.jsonl' \
        -vllm \
        -output
    fi
done