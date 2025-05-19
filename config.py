import transformers
import torch
import torch.nn as nn
import argparse
import os
import sys
import random
import datasets
from datasets import load_dataset, load_from_disk
import pickle
import numpy as np
import json
import functools
import deepspeed
import logging
import jsonlines
import glob
import copy
import time
from datetime import datetime
from itertools import chain
from pathlib import Path
from tqdm import tqdm
from typing import List, Mapping, Optional, Dict
import torch
import torch.nn.functional as F
import torch.distributed as dist
from deepspeed.runtime.zero import GatheredParameters
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.module_inject.replace_policy import BLOOMLayerPolicy
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    DistributedSampler
)
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    GenerationConfig,
    AutoTokenizer,
    AutoConfig,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    Trainer,
    Seq2SeqTrainer,
    BitsAndBytesConfig
)

# from optimum.bettertransformer import BetterTransformer
from scipy.optimize import curve_fit

CONFIG = {
    'batch_size': 256,
    'micro_batch_size': 8,
    'eval_batch_size':16,
    'output_dir': './temp',
    'accumulation_steps': 1,
    'epochs' : 4,
    'max_steps': -1,
    'max_len' : 512,
    'max_new_tokens': 1024,
    'num_beams': 1,
    'learning_rate' : 1e-5,
    'warmup_steps': 16,
    'student_model_path':'',
    'val_data' : '/data/work/processed_cuge_datasets/C3/C3_valid.json',
    'save_steps': 1024,
    'eval_steps' : 102400,
    'val_set_size': 0,
    'train_data' : '/data/work/processed_cuge_datasets/C3/C3_train.json',
    'pretrain_data' : '',
    'model_name': '',
    'model_path' : '',
    'temperature': 3,
    'loss_type': 'topk_normed_ce',
    'alpha': 1e-4,
    'sparsity_pool': 0.1,
    'sparsity_noise': 0.1,
    'lm_loss_weight': 0,
    'loss_weights_2': 0,
    'loss_weights_3': 0,
    'deepspeed_config': None,
    'pad_token_id': 0,
    'num_relation_head': 64,

}
