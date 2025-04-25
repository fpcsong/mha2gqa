#! /bin/bash

ROOT_PATH=/mnt/data/songxiaohui
export HF_HOME=$ROOT_PATH/cache
export NCCL_DEBUG=WARN

MODEL=$1
MODEL_NAME=$2
TOKENIZER_NAME=$3
CONFIG=$4
MXTP=$5
PRUNE_RATIO=$6
WARMUP_RATIO=$7
DESC=$8
LOSS_TYPE=normed_ce

SCHEDULER=cosine
LR=2e-5
LAGLR=2e-5
POOL=0.01
ALPHA=1e-4
BSZ=64

BASEMODEL=$ROOT_PATH/models/$MODEL
TEACHER=$ROOT_PATH/teachers/$MODEL_NAME-benchmark
BENCHMARK_PATH=./benchmarking/full
TRAINDATA=$BENCHMARK_PATH
EVALDATA=$BENCHMARK_PATH

OUTDIR=$ROOT_PATH/results/c3/l0prune/$MODEL/$DESC

for CKPT in $OUTDIR/*
do
    if [ -f $CKPT/config.json ]; then
        mkdir -p $CKPT/pruned
        mkdir -p $CKPT/sft_pruned
        python -m scripts.cvt_prune_weight --model_path $CKPT --model_name $MODEL_NAME --prune_config $CONFIG
        
        deepspeed --hostfile /etc/mpi/hostfile train.py \
            -te \
            -l0_config $CONFIG \
            -t_data $TRAINDATA \
            -v_data $EVALDATA \
            --model_path $CKPT/pruned \
            --model_name $MODEL_NAME \
            --tokenizer_name $TOKENIZER_NAME \
            --bf16 \
            -stage 1 \
            -output \
            -output_dir $OUTDIR \
            -max_len 2048 \
            -m_bsz 1 -e_bsz 4 \
            --warmup_steps 32 \
            --eval_steps 20480 \
            --save_steps $MXTP \
            --lr_scheduler $SCHEDULER \
            --max_steps $MXTP \
            --loss_type $LOSS_TYPE \
            --max_new_tokens 64 \
            --template_name none \
            --prune \
            -lr $LR \
            --bild_topk 16 \
            -attn_impl flash_attention_2 \
            -lag_lr $LAGLR \
            -bsz $BSZ \
            --prune_step_ratio $PRUNE_RATIO \
            --lagrangian_warmup_ratio $WARMUP_RATIO \
            --train_files_pattern  '/c3/train/*.jsonl' \
            --val_files_pattern '/c3/eval/*.jsonl' \
            --alpha $ALPHA \
            --sparsity_pool $POOL \
            --streaming \
            --deepspeed true \
            2>&1 | tee $OUTDIR/tee.log
    fi
done