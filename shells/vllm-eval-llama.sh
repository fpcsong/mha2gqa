#! /bin/bash

ROOT_PATH=/mnt
export HF_HOME=$ROOT_PATH/cache

CKPT_PATH=/workspace/mha2gqa/results/sft/Sheared-LLaMA-1.3B/teachers3epoch
MODEL_NAME=llama
TOKENIZER_NAME=llama

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint_path)
            CKPT_PATH="$2"
            shift # past argument
            shift # past value
            ;;
        -*|--*)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

OUTDIR=$CKPT_PATH/
BENCHMARK_PATH=./benchmarking/datasets

mkdir -p $OUTDIR
TRAINDATA=$BENCHMARK_PATH
EVALDATA=$BENCHMARK_PATH
for BASEMODEL in $CKPT_PATH/*
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
        -max_len 4096 \
        --template_name none \
        --val_files_pattern '/*/eval/input*.jsonl' \
        --train_files_pattern  '/*/train/input*.jsonl' \
        -output \
        -vllm \
        --max_new_tokens 512 \
        2>&1 | tee $OUTDIR/tee.log
    fi
done
