import json

# # 读取原始jsonl文件
# input_file = '/mnt/data/JinQingyun/llm-workspace-sxh/benchmarking/datasets/mmlu/eval/mmlu.jsonl'  # 输入文件名
# output_file = '/mnt/data/JinQingyun/llm-workspace-sxh/benchmarking/datasets/mmlu/eval/ppl_mmlu.jsonl'  # 输出文件名

# with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
#     for line in infile:
#         item = json.loads(line.strip())
#         question = item["question"]
#         choices = item["choices"]
#         answer_index = item["answer"]

#         # 创建新的格式
#         new_format = {
#             "instruction": f"Human:Question: {question}\nChoices: A.{choices[0]} B.{choices[1]} C.{choices[2]} D.{choices[3]}\n Answer:\nAssistant:",
#             "input": "",
#             "output": answer_index,
#             "choices":choices
#         }

#         # 将新格式写入输出文件
#         outfile.write(json.dumps(new_format, ensure_ascii=False) + '\n')

import pandas as pd

# 读取 Parquet 文件
df = pd.read_parquet('/mnt/data/JinQingyun/llm-workspace-sxh/benchmarking/datasets/mmlu/train/validation-00000-of-00001.parquet')

# 写入 JSONL 文件
df.to_json('/mnt/data/JinQingyun/llm-workspace-sxh/benchmarking/datasets/mmlu/train/mmlu_valid.jsonl', orient='records', lines=True)