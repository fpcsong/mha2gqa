# -- coding: utf-8 --**

import argparse
import torch
import os
from tqdm import tqdm
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from prune_utils.composer_deepseek import DeepSeekForL0Prune

def main(args):
    print(args.model_path)
    pruned_model_path = os.path.join(args.model_path, 'pruned')
    config_path = os.path.join(pruned_model_path, 'config.json')
    if os.path.exists(config_path):
        return
    # internlm2 is converted into llama for pruning
    if args.model_name in ['llama', 'deepseek', 'internlm2']:
        model = DeepSeekForL0Prune.from_pretrained(args.model_path)
    model.load_l0_parameters(args.model_path)
    model.convert_format()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.save_pretrained(pruned_model_path)
    model.save_pretrained(pruned_model_path, safe_serialization=True)
    # must after model saving, because arch will reset in save_pretrained
    model.save_config(pruned_model_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--model_path', type=str, required=True,\
                        default='', help='the folder contains model weights')
    parser.add_argument('--model_name', type=str, required=True,\
                        default='', help='model name')
    parser.add_argument('--prune_config', type=str, \
                        default='/mnt/data/songxiaohui/llm-workspace/prune_utils/config/deepseek-4b-30layers.yaml', help='')
    parser.add_argument('--attn_implementation', '-attn', default='flash_attention_2')
    args = parser.parse_args()
    args_dict = vars(args)
    l0_path = os.path.join(args.model_path, 'l0_config.yaml')
    os.system('cp {} {}'.format(args.prune_config, l0_path))
    main(args)
