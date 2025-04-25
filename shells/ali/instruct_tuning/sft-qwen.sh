#! /bin/bash

ROOT_PATH=/mnt/data/songxiaohui
export HF_HOME=$ROOT_PATH/cache

MODEL=$1
MODEL_NAME=$2
TOKENIZER_NAME=$3
DESC=$4
BSZ=$5
MXTP=$6
MBSZ=4
OUTDIR=/mnt/data/JinQingyun/results/sft/$MODEL/$DESC
BASEMODEL=$ROOT_PATH/models/$MODEL
BENCHMARK_PATH=./benchmarking/qwentitle
TRAINDATA=$BENCHMARK_PATH
EVALDATA=$BENCHMARK_PATH
sudo mkdir -p $OUTDIR

deepspeed --num_gpus 8 benchmark.py \
    -it \
    -t_data $TRAINDATA \
    -v_data $EVALDATA \
    --model_path $BASEMODEL \
    --model_name $MODEL_NAME \
    --tokenizer_name $TOKENIZER_NAME \
    --bf16 \
    --stage 1 \
    -output \
    -output_dir $OUTDIR \
    -m_bsz $MBSZ \
    -e_bsz $MBSZ \
    -max_len 4096 \
    --max_new_tokens 128 \
    --eval_steps 204800 \
    --save_steps 204800 \
    --max_steps  $MXTP \
    --template_name none \
    -lr 1e-5 \
    -bsz $BSZ \
    --train_files_pattern  '/train/input*.jsonl' \
    --val_files_pattern '/eval/input*.jsonl' \
    --gradient_checkpointing \
    --deepspeed true \
    #--streaming \
    2>&1 | tee $OUTDIR/tee.log
