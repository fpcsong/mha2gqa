import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import argparse
from torch.nn import CrossEntropyLoss
import vllm
import numpy as np
import json
import math

from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser(description='train/eval a model')
parser.add_argument('--model_path', type=str,default="/mnt/data/JinQingyun/models/contrast/llama-2-8groups-mse-value", help='the name of target llm model')# #/llama-2-7b-hf
parser.add_argument('--val_files_pattern', '-val_files_pattern', type=str, default="/mnt/data/JinQingyun/datasets/filtered_output.jsonl")
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

# 构建所有问题+回答的序列
sequences = []
answers=[]
#for data in dataset[:12]:
    # long_prompt = f"Human:The following is a multiple choice question about {' '.join(data['subject'].split('_'))}.\n\n"
    # question = data["question"]
    # choices = data["choices"]
    # ans=['A','B','C','D']#answer=(data["answer"])
    # long_prompt = f"{long_prompt}{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:\nAssistant: "  
    # answers.append(data["answer"])
sequences.append("tell me a long story")

sampling_params = vllm.SamplingParams(top_k=-1,prompt_logprobs=1,max_tokens=4096)
outputs = llm.generate(sequences, sampling_params, use_tqdm=True)  # 批量传入序列
print(outputs[0].outputs[0].text)
# perplexity=[]
# corrects=0
# for i in outputs:
#     # if i.prompt_logprobs[-1][i.prompt_token_ids[-1]].rank==1:
#     #     corrects+=1
#     #perplexity.append(i.prompt_logprobs[-1][i.prompt_token_ids[-1]].logprob)
    
#     log_prob=i.prompt_logprobs[1:]
#     perplexities=[k[j].logprob for k,j in zip(log_prob,i.prompt_token_ids[1:])]
#     perplexity.append(math.exp(-sum(perplexities)/len(perplexities)))
#     # log_prob=i.prompt_logprobs[-50:]
#     # perplexities=[k[j].logprob for k,j in zip(log_prob,i.prompt_token_ids[1:])]
#     # perplexity.append(math.exp(-sum(perplexities)/len(perplexities)))

# print("correctness:",sum(perplexity)/len(perplexity))




