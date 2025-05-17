#! /bin/bash

ROOT_PATH=/mnt/data/group/songxiaohui
export HF_HOME=$ROOT_PATH/cache

MODEL='Sheared-LLaMA-1.3B'
MODEL_NAME='llama'
TOKENIZER_NAME='llama'
DESC='teachers3epoch'
BSZ=128
MXTP=3459
MBSZ=8
OUTDIR=$ROOT_PATH/results/jqy/sft/$MODEL/$DESC
BASEMODEL=$ROOT_PATH/models/$MODEL
BENCHMARK_PATH=./benchmarking/datasets
TRAINDATA=$BENCHMARK_PATH
EVALDATA=$BENCHMARK_PATH
sudo mkdir -p $OUTDIR
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
deepspeed --hostfile /etc/mpi/hostfile benchmark.py \
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
    --train_files_pattern  '/*/train/input_*.jsonl' \
    --val_files_pattern '/*/eval/input*.jsonl' \
    --gradient_checkpointing \
    --deepspeed true \
    #--streaming \
    2>&1 | tee $OUTDIR/tee.log
