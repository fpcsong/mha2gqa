#! /bin/bash

ROOT_PATH=/mnt/data/JinQingyun
export HF_HOME=$ROOT_PATH/cache
export NCCL_DEBUG=WARN
MODEL=$1
MODEL_NAME=$2
CONFIG=$3
LR=$4
MXTP=$5
TOPK=$6
WARMUP_RATIO=$7
DESC=$8
TEACHER=$9
TEMP=${10:-3.0}
LOSS_TYPE=${11:-topk_normed_ce}
BSZ=${12:-64}
SEED=${13:-19960301}

SCHEDULER=cosine
PRUNE_RATIO=0.8
SFTTP=1024
POOL=0.01
ALPHA=1e-3
LAGLR=$LR
TOKENIZER_NAME=$MODEL_NAME
BASEMODEL=$ROOT_PATH/models/$MODEL
TEACHER=$ROOT_PATH/models/$TEACHER
BENCHMARK_PATH=./benchmarking/datasets
TRAINDATA=$BENCHMARK_PATH
EVALDATA=$BENCHMARK_PATH

OUTDIR=$ROOT_PATH/results/benchmark/l0prune/distil/$MODEL/$DESC

mkdir -p $OUTDIR

rm -rf $OUTDIR/*

# accelerate launch train.py \
deepspeed --hostfile /etc/mpi/hostfile train.py \
    -it \
    -l0_config $CONFIG \
    -t_data $TRAINDATA \
    -v_data $EVALDATA \
    --model_path $BASEMODEL \
    --model_name $MODEL_NAME \
    --tokenizer_name $TOKENIZER_NAME \
    --teacher_model_path $TEACHER \
    --bf16 \
    -output \
    -output_dir $OUTDIR \
    -max_len 2048 \
    -m_bsz 1 -e_bsz 4 \
    --loss_type $LOSS_TYPE \
    --bild_topk $TOPK \
    --temperature $TEMP \
    --warmup_steps 32 \
    --eval_steps 204800 \
    --save_steps $MXTP \
    --lr_scheduler $SCHEDULER \
    --max_steps $MXTP \
    --max_new_tokens 64 \
    --template_name none \
    --prune \
    --lm_loss_weight 0.0 \
    --gradient_checkpointing \
    --stage 1 \
    -lr $LR \
    -lag_lr $LAGLR \
    -bsz $BSZ \
    --prune_step_ratio $PRUNE_RATIO \
    --lagrangian_warmup_ratio $WARMUP_RATIO \
    --train_files_pattern  '/*/train/input*.jsonl' \
    --val_files_pattern '/*/eval/input*.jsonl' \
    --alpha $ALPHA \
    --sparsity_pool $POOL \
    --deepspeed true \
    --streaming \
    --attn_impl eager \
    #--do_mid_distil \
    
    --seed $SEED
    2>&1 | tee $OUTDIR/tee.log

for CKPT in $OUTDIR/*
do
    if [ -f $CKPT/config.json ]; then
        mkdir -p $CKPT/pruned
        mkdir -p $CKPT/sft_pruned
        cp $BASEMODEL/*.py $CKPT/
        cp $BASEMODEL/*.py $CKPT/pruned/
        python -m scripts.cvt_prune_weight --model_path $CKPT --model_name $MODEL_NAME --prune_config $CONFIG
        
        python benchmark.py -te -v_data $EVALDATA \
        --model_path $CKPT/pruned \
        --model_name $MODEL_NAME \
        --bf16 \
        -output_dir $OUTDIR \
        -m_bsz 2 \
        -e_bsz 2 \
        -max_len 2048 \
        --template_name none \
        --val_files_pattern '/*/eval/*.jsonl' \
        --train_files_pattern  '/*/train/*.jsonl' \
        --eval_vllm \
        -output
    fi
done