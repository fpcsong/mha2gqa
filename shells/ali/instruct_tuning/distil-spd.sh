#! /bin/bash

ROOT_PATH=/mnt/data/songxiaohui
export HF_HOME=$ROOT_PATH/cache

MODEL=$1
MODEL_NAME=$2
TOKENIZER_NAME=$3
LR=$4
MXTP=$5
TRAIN=$6
LOSS_TYPE=$7
DESC=$8
TEACHER=$9
MBSZ=${10:-2}
TEMP=${11:-3.0}
LMW=${12:-0}
BSZ=128
OUTDIR=$ROOT_PATH/results/summary/$MODEL/$DESC
BASEMODEL=$ROOT_PATH/models/$MODEL
TEACHERMODEL=$ROOT_PATH/teachers/$TEACHER
TRAINDATA=$ROOT_PATH/data/summary_datasets/train/$TRAIN
EVALDATA=$ROOT_PATH/data/summary_datasets/train/eval
sudo mkdir -p $OUTDIR
rm -r $OUTDIR/*
deepspeed --hostfile /etc/mpi/hostfile train.py \
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
    -max_len 4096 \
    --warmup_steps 640 \
    --eval_steps 204800 \
    --save_steps 5120 \
    --distil \
    --teacher_model_path $TEACHERMODEL \
    --lm_loss_weight $LMW \
    --bild_topk 16 \
    --temperature $TEMP \
    --max_steps  $MXTP \
    --lr_scheduler cosine \
    --loss_type $LOSS_TYPE \
    --temperature $TEMP \
    --template_name none \
    -lr $LR \
    -bsz $BSZ \
    --gradient_checkpointing \
    --streaming \
    -output \
    --stage 2\
    --deepspeed ds_config3_no_offload.json \
    2>&1 | tee $OUTDIR/tee.log
