import datasets
from datasets import load_dataset
import os
import argparse
import glob
import random
import pandas as pd
from difflib import SequenceMatcher
from tqdm import tqdm
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def add_sort_key(batch):
    batch_size = len(batch['input'])
    batch['key'] = []
    for i in range(batch_size):
        batch['key'].append(batch['instruction'][i] + batch['input'][i])
    return batch
def main(args):
    data_files = glob.glob(args.data_path+args.files_pattern, recursive=True)
    origin_datasets = load_dataset("json", 
                                data_files=data_files,
                                num_proc=96,
                                split='train',
                                )
    origin_datasets = origin_datasets.map(add_sort_key,
                                          batched=True,
                                          batch_size=args.batch_size,
                                          num_proc=96).sort('key')
    cleaned_datasets = []
    for idx, item in tqdm(enumerate(origin_datasets)):
        if idx > 1 and similar(item['key'], origin_datasets[idx-1]['key']) > 0.9:
            continue
        cleaned_datasets.append(item)
    cleaned_datasets = datasets.Dataset.from_pandas(pd.DataFrame(data=cleaned_datasets))
    cleaned_datasets = cleaned_datasets.remove_columns('key')

    os.makedirs(args.output_path, exist_ok=True)
    
    for idx in range(args.num_shards):
        subset = cleaned_datasets.shard(num_shards=args.num_shards, index=idx)
        subset.to_json(os.path.join(args.output_path, 'data-{}.jsonl'.format(idx)),
                                batch_size=args.batch_size,
                                num_proc=96,
                                lines=True, force_ascii=False
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,\
                        default='', help='the folder contains data files')
    parser.add_argument('--output_path', type=str, required=True,\
                        default='', help='the folder contains target data files')
    parser.add_argument('--batch_size', '-bsz', type=int, default=4096, required=False)
    parser.add_argument('--num_shards', '-num_shards', type=int, default=1, required=False)
    parser.add_argument('--files_pattern', '-files_pattern', type=str, default='/*.jsonl')
    args = parser.parse_args()
    main(args)
