#! /bin/bash

ROOT_PATH=/mnt
export HF_HOME=$ROOT_PATH/cache
export NCCL_DEBUG=WARN

MODEL_NAME='llama'
TOKENIZER_NAME='llama'
DESC='teachers2epoch'
BSZ=128
MXTP=2306
MBSZ=4

BASEMODEL=/Sheared-LLaMA-1.3B
BENCHMARK_PATH=./benchmarking/datasets
TRAINDATA=$BENCHMARK_PATH
EVALDATA=$BENCHMARK_PATH
NUM_GPUS=8

while [[ $# -gt 0 ]]; do
    case $1 in
        --base_model_path)
            MODELPATH="$2"
            shift # past argument
            shift # past value
            ;;
        --train_steps)
            MXTP="$2"
            shift
            shift
            ;;
        --batch_size)
            BSZ="$2"
            shift
            shift
            ;;
        --experiment_name)
            DESC="$2"
            shift
            shift
            ;;
        --seed)
            SEED="$2"
            shift
            shift
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift
            shift
            ;;
        -*|--*)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

BASEMODEL=$(basename "$MODELPATH") # ${MODELPATH##*/}

OUTDIR=$ROOT_PATH/results/sft/$BASEMODEL/$DESC

mkdir -p $OUTDIR
rm -rf $OUTDIR/*

deepspeed --num_gpus $NUM_GPUS benchmark.py \
    -it \
    -t_data $TRAINDATA \
    -v_data $EVALDATA \
    --model_path $MODELPATH \
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
    --train_files_pattern  '/*/train/input*.jsonl' \
    --val_files_pattern '/*/eval/input*.jsonl' \
    --gradient_checkpointing \
    --deepspeed true \
    #--streaming \
    2>&1 | tee $OUTDIR/tee.log
