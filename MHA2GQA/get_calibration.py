from config import *
from utils import *
from prompter import Prompter
from tokenize_functions import *
from benchmarking.evaluation import evaluation_func
from ds_utils import get_train_ds_config
from fsdp_utils import get_train_fsdp_config
from trainers.custom_trainer import *
from callbacks import *
from prune_utils.composer_deepseek import DeepSeekForL0Prune
from transformers import LlamaConfig, LlamaModel

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                                    max_length=CONFIG['max_len'],
                                                    padding_side="right",
                                                    truncation_side="right",
                                                    trust_remote_code=True,
                                                    use_fast=True)
    
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    if args.tokenizer_name in ['llama', 'baichuan', 'cpm']:
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 0
        tokenizer.unk_token_id = 0

    tokenizer.add_special_tokens = False

    model_dtype = torch.float16 if args.fp16 else torch.bfloat16

    model = LlamaModel(config)