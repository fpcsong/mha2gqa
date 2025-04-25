#! /bin/bash

ROOT_PATH=/mnt/data/songxiaohui
export HF_HOME=$ROOT_PATH/cache
export NCCL_DEBUG=WARN
MODEL=$1
MODEL_NAME=$2
TOKENIZER_NAME=$3
TEACHER=$4
CONFIG=$5
TRAIN=$6
DESC=$7
PRUNE_RATIO=0.8
WARMUP_RATIO=0.2
BSZ=128
MXTP=51200
ALPHA=1e-3
BASEMODEL=$ROOT_PATH/models/$MODEL
TEACHER_MODEL_PATH=$ROOT_PATH/teachers/$TEACHER
TRAINDATA=$ROOT_PATH/data/summary_datasets/train/$TRAIN
EVALDATA=$ROOT_PATH/data/summary_datasets/train/eval

OUTDIR=$ROOT_PATH/results/summary/l0prune/$MODEL/$DESC

mkdir -p $OUTDIR

rm -rf $OUTDIR/*
deepspeed --num_gpus 8 train.py \
    -it \
    -l0_config $CONFIG \
    -t_data $TRAINDATA \
    -v_data $EVALDATA \
    --model_path $BASEMODEL \
    --model_name $MODEL_NAME \
    --teacher_model_path $TEACHER_MODEL_PATH \
    --tokenizer_name $TOKENIZER_NAME \
    --bf16 \
    -output \
    -output_dir $OUTDIR \
    -max_len 4096 \
    -m_bsz 1 -e_bsz 4 \
    --warmup_steps 1024 \
    --eval_steps 1000000 \
    --save_steps $MXTP \
    --lr_scheduler cosine \
    --max_steps $MXTP \
    --template_name none \
    --streaming \
    --temperature 3.0 \
    --prune_step_ratio $PRUNE_RATIO \
    --lagrangian_warmup_ratio $WARMUP_RATIO \
    --prune \
    --gradient_checkpointing \
    --stage 2\
    -lr 2e-5 \
    -lag_lr 2e-5 \
    -bsz $BSZ \
    --alpha $ALPHA \
    --sparsity_pool 0.01 \
    --deepspeed true \
    2>&1 | tee $OUTDIR/tee.log


for CKPT in $OUTDIR/*
do
    if [ -f $CKPT/config.json ]; then
        mkdir -p $CKPT/pruned
        mkdir -p $CKPT/sft_pruned
        cp $BASEMODEL/*.py $CKPT/
        cp $BASEMODEL/*.py $CKPT/pruned/
        python -m scripts.cvt_prune_weight --model_path $CKPT --model_name $MODEL_NAME --prune_config $CONFIG

        deepspeed --num_gpus 8 train.py \
            -it \
            -t_data $TRAINDATA \
            -v_data $EVALDATA \
            --model_path $CKPT/pruned \
            --model_name $MODEL_NAME \
            --tokenizer_name $TOKENIZER_NAME \
            --teacher_model_path $TEACHER_MODEL_PATH \
            --bf16 \
            -stage 1 \
            --distil \
            --gradient_checkpointing \
            --temperature 3.0 \
            -output \
            -output_dir $CKPT/sft_pruned \
            -max_len 4096 \
            -m_bsz 1 -e_bsz 8 \
            --warmup_steps 32 \
            --eval_steps 2048000 \
            --save_steps 5120 \
            --lr_scheduler cosine \
            --max_steps $MXTP \
            --template_name none \
            -lr 2e-5 \
            -bsz $BSZ \
            --deepspeed true \
            --streaming \
            2>&1 | tee $CKPT/sft_pruned/tee.log
    fi
done