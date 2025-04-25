#! /bin/bash

ROOT_PATH=/mnt/data/songxiaohui
export HF_HOME=$ROOT_PATH/cache

MODEL=$1
MODEL_NAME=$2
TOKENIZER_NAME=$3
TEACHER=$4
TRAIN=$5
DESC=$6
MXTP=${7:-10240}
LMW=${8:-0}
LOSS_TYPE=${9:-topk_normed_ce}
TEMP=${10:-2.0}
LORA_RANK=32
LORA_ALPHA=32
BSZ=64
LR=1e-4
BASEMODEL=$ROOT_PATH/models/$MODEL
TEACHERMODEL=$ROOT_PATH/teachers/$TEACHER
OUTDIR=$ROOT_PATH/results/summary/$MODEL/$DESC
TRAINDATA=$ROOT_PATH/data/summary_datasets/train/$TRAIN
EVALDATA=$ROOT_PATH/data/summary_datasets/train/eval

mkdir -p $OUTDIR
rm -rf $OUTDIR/*

deepspeed --hostfile /etc/mpi/hostfile train.py \
    -it \
    -t_data $TRAINDATA \
    -v_data $EVALDATA \
    --model_path $BASEMODEL \
    --model_name $MODEL_NAME \
    --tokenizer_name $TOKENIZER_NAME \
    --distil \
    --teacher_model_path $TEACHERMODEL \
    --lm_loss_weight $LMW \
    --bild_topk 16 \
    --temperature $TEMP \
    --bf16 \
    -output_dir $OUTDIR \
    -m_bsz 1 \
    -e_bsz 1 \
    -max_len 4096 \
    --warmup_steps 64 \
    --eval_steps 204800 \
    --save_steps 1024 \
    --max_steps $MXTP \
    --streaming \
    --loss_type $LOSS_TYPE \
    --lr_scheduler cosine \
    --template_name none \
    -lr $LR \
    -bsz $BSZ \
    --gradient_checkpointing \
    -output \
    --stage 1\
    --lora \
    --lora_r $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --deepspeed ds_config3_no_offload.json \
    2>&1 | tee $OUTDIR/tee.log
