#! /bin/bash

ROOT_PATH=/mnt/data/songxiaohui
export HF_HOME=$ROOT_PATH/cache

MODEL=$1
MODEL_NAME=$2
TOKENIZER_NAME=$3
TRAIN=$4
DESC=$5
MBSZ=1
BSZ=128
OUTDIR=$ROOT_PATH/results/summary/$MODEL/$DESC
BASEMODEL=$ROOT_PATH/models/$MODEL
TRAINDATA=$ROOT_PATH/data/summary_datasets/train/$TRAIN
EVALDATA=$ROOT_PATH/data/summary_datasets/train/eval
sudo mkdir -p $OUTDIR

deepspeed --num_gpus 8 train.py -it \
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
    --warmup_steps 64 \
    --eval_steps 204800 \
    --save_steps 2048 \
    --max_steps  51200 \
    --lr_scheduler cosine \
    --template_name none \
    -lr 5e-5 \
    -bsz $BSZ \
    --gradient_checkpointing \
    --streaming \
    -output \
    --stage 2\
    --deepspeed ds_config3_no_offload.json \
    2>&1 | tee $OUTDIR/tee.log
