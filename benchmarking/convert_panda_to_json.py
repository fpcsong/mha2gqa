import pandas as pd

# 读取 Parquet 文件
parquet_file = '/mnt/data/JinQingyun/llm-workspace-sxh/benchmarking/datasets/mmlu/test-00000-of-00001.parquet'
df = pd.read_parquet(parquet_file)

# 将 DataFrame 转换为 JSON 文件
json_file = '/mnt/data/JinQingyun/llm-workspace-sxh/benchmarking/datasets/mmlu/mmlu.jsonl'
df.to_json(json_file, orient='records', lines=True)

print(f"Converted {parquet_file} to {json_file}")