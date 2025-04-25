from datasets import load_dataset, concatenate_datasets
import os
import argparse
import glob
import random

def align_for_chat(batch):
    batch_size = len(batch['input'])
    for i in range(batch_size):
        if batch['instruction'][i][:10].find("Human") == -1:
            batch['instruction'][i] = "Human:" + batch['instruction'][i]
            batch['input'][i] += '\nAssistant:'
    return batch
def main(args):
    additional_data_path = os.path.join(args.data_path, 'additional')
    summary_data_path = os.path.join(args.data_path, 'summary')
    additional_data_files = glob.glob(additional_data_path + args.files_pattern, recursive=True)
    summary_data_files = glob.glob(summary_data_path + args.files_pattern, recursive=True)
    additional_dataset = load_dataset("json", 
                                data_files=additional_data_files,
                                num_proc=96,
                                split='train',
                                )
    summary_dataset = load_dataset("json",
                                   data_files=summary_data_files,
                                   num_proc=96,
                                   split='train')
    
    additional_dataset = additional_dataset.filter(lambda item: item['output']!= '' \
                                                        and item['output'] is not None \
                                                        and item['instruction'].find('http') == -1 \
                                                        and item['input'].find('http') == -1 \
                                                        and item['output'].find('http') == -1 \
                                                        and item['input'].find('Tiger') == -1 \
                                                        and item['output'].find('Tiger') == -1 \
                                                        and item['input'].find('OpenAI') == -1 \
                                                        and item['output'].find('OpenAI') == -1 \
                                                        ,
                                                       batch_size=args.batch_size,
                                                       num_proc=96)

    additional_dataset = additional_dataset.shuffle().map(align_for_chat,
                                        batched=True,
                                        batch_size=args.batch_size,
                                        num_proc=96)
    summary_dataset = summary_dataset.shuffle().map(align_for_chat,
                                                    batched=True,
                                                    batch_size=args.batch_size,
                                                    num_proc=96)
    num_summary = len(summary_dataset)
    num_additional = int(num_summary * args.alpha)
    num_additional = min(len(additional_dataset)-1, num_additional)
    print('num summary data {}\nnum additional data {}'.format(num_summary, num_additional))
    additional_dataset = additional_dataset.train_test_split(num_additional)['test']
    full_dataset = concatenate_datasets([summary_dataset, additional_dataset])
    full_dataset = full_dataset.shuffle()
    os.makedirs(args.output_path, exist_ok=True)
    
    for idx in range(args.num_shards):
        subset = full_dataset.shard(num_shards=args.num_shards, index=idx)
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
    parser.add_argument('--alpha', '-alpha', type=float, default=2.0, required=False)
    parser.add_argument('--num_shards', '-num_shards', type=int, default=32, required=False)
    parser.add_argument('--files_pattern', '-files_pattern', type=str, default='/**/*.jsonl')
    args = parser.parse_args()
    main(args)
