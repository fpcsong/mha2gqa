import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import argparse
from torch.nn import CrossEntropyLoss
import vllm
import numpy as np
import json
import math
import pandas as pd

from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser(description='train/eval a model')
parser.add_argument('--model_path', type=str,default="/mnt/data/JinQingyun/models/llama-sft", help='the name of target llm model')
parser.add_argument('--val_files_pattern', '-val_files_pattern', type=str, default="/mnt/data/JinQingyun/llm-workspace-sxh/benchmarking/datasets/mmlu/eval/mmlu.jsonl")
args = parser.parse_args()
# 初始化模型
llm = vllm.LLM(
        args.model_path, 
        trust_remote_code=True,
        tensor_parallel_size=1,
        quantization=None,
        enable_chunked_prefill=False,
        # kv_cache_dtype="fp8",
        )



# 存储所有数据的列表
dataset = []
file_path=args.val_files_pattern
# 打开文件并逐行读取
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 解析 JSON 数据并添加到列表
        item = json.loads(line.strip())
        dataset.append(item)


def create_prompt(subject):
    long_prompt = f"The following are multiple choice questions (with answers) about {' '.join(subject.split('_'))}.\n\n"
    filtered_df = []

    with open('/mnt/data/JinQingyun/llm-workspace-sxh/benchmarking/datasets/mmlu/train/mmlu_valid.jsonl', 'r') as infile:
        for line in infile:
            record = json.loads(line)
            if record['subject'] == subject:
                filtered_df.append(record)

    choices = ["A", "B", "C", "D"]
    for shot in range(5):
        question = filtered_df[shot]['question']
        answer = choices[filtered_df[shot]['answer']]
        choice_A = filtered_df[shot]['choices'][0]
        choice_B = filtered_df[shot]['choices'][1]
        choice_C = filtered_df[shot]['choices'][2]
        choice_D = filtered_df[shot]['choices'][3]


        long_prompt = f"{long_prompt}{question}\nA. {choice_A}\nB. {choice_B}\nC. {choice_C}\nD. {choice_D}\nAnswer: {answer}\n\n"
    return long_prompt
# 构建所有问题+回答的序列
sequences = []
answers=[]
pre_prompt={}
for data in dataset:
    if data['subject'] not in pre_prompt:
        pre_prompt[data['subject']]=create_prompt(data['subject'])

    #long_prompt = f"Human:The following is a multiple choice question about {' '.join(data['subject'].split('_'))}.\n\n"
    question = data["question"]
    choices = data["choices"]
    ans=['A','B','C','D']#answer=(data["answer"])
    long_prompt = f"{pre_prompt[data['subject']]}{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:\nAssistant: "  
    answers.append(data["answer"])
    #sequences.extend([f"{long_prompt}{an}" for an in ans])
    sequences.append(long_prompt+ans[data["answer"]])

sampling_params = vllm.SamplingParams(top_k=-1,prompt_logprobs=1,max_tokens=500)
outputs = llm.generate(sequences, sampling_params, use_tqdm=True)  # 批量传入序列
perplexity=[]
corrects=0
for i in outputs:
    if i.prompt_logprobs[-1][i.prompt_token_ids[-1]].rank==1:
        corrects+=1
    #perplexity.append(i.prompt_logprobs[-1][i.prompt_token_ids[-1]].logprob)
    
    # log_prob=i.prompt_logprobs[1:]
    # perplexities=[k[j].logprob for k,j in zip(log_prob,i.prompt_token_ids[1:])]
    # perplexity.append(math.exp(-sum(perplexities)/len(perplexities)))


# chunks = [perplexity[i:i + 4] for i in range(0, len(perplexity), 4)]
# max_positions = [chunk.index(max(chunk)) for chunk in chunks]
# corrects=sum(1 for a, b in zip(max_positions, answers) if a == b)
#print("correctness:",corrects/len(chunks))
print("correctness:",corrects/len(sequences))



