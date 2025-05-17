#! /bin/bash

ROOT_PATH=/mnt
export HF_HOME=$ROOT_PATH/cache
export NCCL_DEBUG=WARN

MODEL_NAME=llama
CONFIG=./prune_utils/config/llama2-1.3B-gqa-8groups.yaml
LR=1e-5
MXTP=13836
TOPK=16
WARMUP_RATIO=0.3
DESC=groups16
TEMP=3.0
LOSS_TYPE=topk_normed_ce
BSZ=64
SEED=19960301

SCHEDULER=cosine
PRUNE_RATIO=0.8
POOL=0.01
ALPHA=1e-3
LAGLR=$LR
TOKENIZER_NAME=llama
MODELPATH=/workspace/Sheared-LLaMA-1.3B
TEACHER=./results/sft/Sheared-LLaMA-1.3B/teachers3epoch/Sheared-LLaMA-1.3B
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
        --prune_config)
            CONFIG="$2"
            shift
            shift
            ;;
        --teacher_path)
            TEACHER="$2"
            shift
            shift
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

BASEMODEL=$(basename "$MODELPATH")
OUTDIR=$ROOT_PATH/results/l0prune/distil/$BASEMODEL/$DESC

mkdir -p $OUTDIR
rm -rf $OUTDIR/*

# accelerate launch train.py \
deepspeed --num_gpus $NUM_GPUS train.py \
    -it \
    -l0_config $CONFIG \
    -t_data $TRAINDATA \
    -v_data $EVALDATA \
    --model_path $MODELPATH \
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
    --attn_impl flash_attention_2 \
    --seed $SEED
    2>&1 | tee $OUTDIR/tee.log

for CKPT in $OUTDIR/*
do
    if [ -f $CKPT/config.json ]; then
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