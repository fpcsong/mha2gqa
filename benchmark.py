# -- coding: utf-8 --**
from config import *
from utils import *
from prompter import Prompter
from tokenize_functions import *
from benchmarking.evaluation import evaluation_func
from ds_utils import get_train_ds_config
from trainers.custom_trainer import *
from callbacks import PeftCallback, ProcessBarCallback


import gc
def get_paramsgroup(model):
    no_decay = ['bias', 'LayerNorm.weight']

    params = []
    warmup_params = []
    for name, param in model.named_parameters():
        # if id(param) in frozen_params:
        #     continue
        lr = CONFIG['learning_rate']
        weight_decay = 0

        if not any(nd in name for nd in no_decay):
            weight_decay = 1e-4
        params.append(
            {
                'params': param,
                'lr': lr,
                'weight_decay': weight_decay
            }
        )
    return params

def log2file(args, msg:str):
    if args.output_predict:
        f = open(args.output_file, 'a')
        f.write(str(msg))
        f.write('\n')
        f.close()

def collate_fn_for_glm(batch):
    print(batch)

def eval_collate_fn(batch):
    # padding=left, labels are kept at the last
    labels_len = [(torch.LongTensor(item['labels']) != -100).long().sum() for item in batch]
    origin_input_ids = [item['input_ids'][:-labels_len[idx]] for idx,item in enumerate(batch)]
    origin_label_ids = [item['input_ids'][-labels_len[idx]:] for idx,item in enumerate(batch)]

    input_strs = CONFIG['tokenizer'].batch_decode(origin_input_ids, 
                                                 skip_special_tokens=True)
    labels = CONFIG['tokenizer'].batch_decode(origin_label_ids, 
                                                 skip_special_tokens=True)
    ret = {}
    ret['labels'] = labels
    ret['input'] = input_strs
    ret['task_name'] = [batch[i]['task_name'] for i in range(len(batch))]
    ret['response_split'] = [batch[i]['response_split'] for i in range(len(batch))]
    inputs = CONFIG['tokenizer'](input_strs,  
                                return_tensors="pt", 
                                padding='longest')
    ret['input_ids'] = inputs['input_ids']
    ret['attention_mask'] = inputs['attention_mask']
    return ret

def response_generation(
    args,
    model,
    data_points,
):
    input_ids = data_points["input_ids"]
    attention_mask = data_points['attention_mask']
    if torch.cuda.device_count() > 0:
        input_ids = input_ids.to(torch.cuda.current_device())
        attention_mask = attention_mask.to(torch.cuda.current_device())
    # return ['?'] * len(prompts)
    # print(input_ids)

    # generation_config_file = os.path.join('generation_configs', args.gen_config)
    # generation_dict = {}
    # with open(generation_config_file, 'r') as f:
    #     for line in f:
    #         k,v = line[:-1].split('=')
    #         generation_dict[k] = eval(v)
    # model.generation_config = GenerationConfig(
    #     ** generation_dict
    # )
    if args.model_name in ['phi']:
        with torch.no_grad():
            generation_output = model.generate(
            input_ids=input_ids,
            return_dict_in_generate=True,
            min_new_tokens=1,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=CONFIG['tokenizer'].pad_token_id
        )
    else:
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                top_p=None,
                top_k=None,
                return_dict_in_generate=True,
                output_scores=True,
                min_new_tokens=1,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=CONFIG['tokenizer'].pad_token_id
            )

    s = generation_output.sequences
    s = s[:, attention_mask.shape[-1]:]
    output = CONFIG['tokenizer'].batch_decode(s, skip_special_tokens=True)
    return output

def train(args, model, trainset, evalset):
    # for benchmarking, train each task from scratch
    model_to_train = copy.deepcopy(model)
    training_ds_config = get_train_ds_config(
        offload=args.offload,
        stage=args.stage,
        enable_hybrid_engine=False,
        inference_tp_size=CONFIG['world_size'],
        # use_qat=args.moq,
        max_out_tokens=2048
        )

    collate_fn = DataCollatorForSeq2Seq(CONFIG['tokenizer'],
                                        pad_to_multiple_of=8,
                                        return_tensors="pt",
                                        padding='longest')
    # training_ds_config['scheduler']['params']['warmup_num_steps'] = CONFIG['warmup_steps']
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        warmup_steps=CONFIG['warmup_steps'],
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        optim="adamw_torch",
        lr_scheduler_type='cosine',
        max_grad_norm=1.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=1e-4,
        bf16=args.bf16,
        logging_steps=10,
        log_on_each_node=False,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        load_best_model_at_end=False,
        save_steps = args.save_steps,
        save_total_limit=10,
        # jit_mode_eval=True,
        report_to='none',
        dataloader_num_workers=os.cpu_count(),
        deepspeed=training_ds_config if args.deepspeed else None,
        ddp_find_unused_parameters=False
    )
    
    trainer = CustomTrainerForSFT(
        config=CONFIG,
        model=model_to_train,
        train_dataset=trainset,
        eval_dataset=evalset,
        tokenizer=CONFIG['tokenizer'],
        args=training_args,
        data_collator=collate_fn,
        callbacks=[ProcessBarCallback]
    )
    if hasattr(trainer.accelerator, 'dataloader_config'):
        trainer.accelerator.dataloader_config.dispatch_batches=False
    else:
        trainer.accelerator.dispatch_batches=False        
    trainer.train()
    return model_to_train

def evaluate(args, model, evalset, all_metrics, training_flags=False):
    CONFIG['tokenizer'].padding_side = "left"
    model.eval()
    # if model.device == torch.device('cpu') and torch.cuda.device_count() > 0:
    #     model = model.to(torch.cuda.current_device())
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ds_quant_config = deepspeed.inference.config.QuantizationConfig()
    ds_quant_config.enabled = True
    ds_quant_config.activation.q_groups = 1
    ds_quant_config.activation.q_type = 'asymmetric'
    ds_quant_config.activation.num_bits = 16
    ds_quant_config.weight.num_bits = 8
    ds_quant_config.weight.q_type = 'asymmetric'

    ds_config = {
        "replace_with_kernel_inject": True,
        "tensor_parallel": {
            "enabled": True,
            "tp_size": world_size
        },
        # "quant": ds_quant_config
    }
    if args.deepspeed and not training_flags:
        ds_engine = deepspeed.init_inference(
                                    model,
                                    dtype=torch.half,
                                    config=ds_config
                                    )
        model = ds_engine.module
    elif not args.deepspeed:
        model.to('cuda:0')
    
    def add_label_length(example):
        example['label_length'] = len(example['labels'])
        return example
    evalset = evalset.map(add_label_length)
    # 根据 'label_length' 进行排序
    evalset = evalset.sort('label_length', reverse=True)

    sampler = SequentialSampler(evalset)
    eval_dataloader = DataLoader(evalset,
                                batch_size=args.eval_batch_size,
                                collate_fn=eval_collate_fn,
                                sampler=sampler)
    labels = {}
    predicts = {}
    inputs = {}
    for examples in tqdm(eval_dataloader, desc='evaluating', \
                        disable=args.local_rank not in [0, -1]):
        outputs = response_generation(args, model, examples)

        # if args.local_rank in [0, -1]:
        for idx, label in enumerate(examples['labels']):
            task_name = examples['task_name'][idx]
            response_split = examples['response_split'][idx]
            if labels.get(task_name) is None:
                labels[task_name] = []
                predicts[task_name] = []
                inputs[task_name] = []
            inputs[task_name].append(examples['input'][idx])
            if response_split != '':
                labels[task_name].append(label.split(response_split)[-1])
                predicts[task_name].append(outputs[idx].split(response_split)[-1])
            else:
                labels[task_name].append(label)
                predicts[task_name].append(outputs[idx])
    # calc metrics
    task_res = []
    for task_name in labels.keys():
        for l, p, i in zip(labels[task_name], predicts[task_name], inputs[task_name]):
            task_res.append(
                {
                    'task_name': task_name,
                    'input': i,
                    'label': l,
                    'predict': p,
                }
            )
    results_path = args.output_file.replace('txt', 'jsonl')
    with jsonlines.Writer(open(results_path, 'w', encoding='utf-8')) as writer:
        writer.write_all(task_res)
    numbers = []
    all_metrics = {}
    for task_name in labels.keys():
        res = evaluation_func(task_name, labels[task_name], predicts[task_name], all_inputs=inputs[task_name])
        all_metrics[res['task_name']] = res['result']
        numbers.append(res['result'])
        if args.local_rank in [0, -1]:
            print(res)
    if args.local_rank in [0, -1]:
        print(sum(numbers)/len(numbers))
        all_metrics['z_mean'] = sum(numbers)/len(numbers)
        log2file(args, json.dumps(all_metrics, sort_keys=True))

def vllm_eval(args):
    
    if args.vllm_quant == "8":
        quantization = "MixQ8bit"
    elif args.vllm_quant == "4":
        quantization = "MixQ4bit"    
    elif args.vllm_quant == "awq":
        quantization = "AWQ"
    elif args.vllm_quant == 'bnb':
        quantization = 'bitsandbytes'
    else:
        quantization = None
    print0('quant:', quantization, args.vllm_quant)
    import vllm
    llm = vllm.LLM(
        args.model_path, 
        trust_remote_code=True,
        tensor_parallel_size=args.num_gpus,
        quantization=quantization,
        enable_chunked_prefill=False,
        # kv_cache_dtype="fp8",
        )
    sampling_params = vllm.SamplingParams(top_k=1, max_tokens=args.max_new_tokens)
    val_file_names = args.val_data
    all_metrics = {}
    batched = True
    val_file_names = glob.glob(args.val_data+args.val_files_pattern, recursive=True)
    val_file_names = list(filter(lambda item: 'mmlu' not in item, val_file_names))
    print('mmlu is not considered')
    eval_dataset = load_dataset("json", 
                            data_files=val_file_names,
                            split='train',
                            streaming=args.streaming,
                            )
    sampler = RandomSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                batch_size=args.eval_batch_size,
                                sampler=sampler)
    labels = []
    predicts = []
    inputs = []
    task_names = []
    for examples in tqdm(eval_dataloader, desc='evaluating', \
                        disable=args.local_rank not in [0, -1]):
        for idx, label in enumerate(examples['output']):
            task_names.append(examples['task_name'][idx])
            response_split = examples['response_split'][idx]
            inputs.append(examples['instruction'][idx] + examples['input'][idx])
            if response_split != '':
                labels.append(label.split(response_split)[-1])
            else:
                labels.append(label)
    predicts = llm.generate(inputs, sampling_params=sampling_params, use_tqdm=True)
    predicts = [item.outputs[0].text.strip() for item in predicts]
    all_labels = {}
    all_predicts = {}
    all_inputs = {}
    for i,l,p,n in zip(inputs, labels, predicts, task_names):
        if n not in all_labels:
            all_labels[n] = []
            all_predicts[n] = []
            all_inputs[n] = []
        all_inputs[n].append(i)
        all_labels[n].append(l)
        all_predicts[n].append(p)
    task_res = []
    for task_name in all_labels.keys():
        for l, p, i in zip(all_labels[task_name], all_predicts[task_name], all_inputs[task_name]):
            task_res.append(
                {
                    'task_name': task_name,
                    'input': i,
                    'label': l,
                    'predict': p,
                }
            )
    results_path = args.output_file.replace('txt', 'jsonl')
    if args.local_rank in [0, -1]:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
        with jsonlines.Writer(open(results_path, 'w', encoding='utf-8')) as writer:
            writer.write_all(task_res)
            writer.close()
        numbers = 0
        total=0
        for task_name in all_labels.keys():
            res = evaluation_func(task_name, all_labels[task_name], all_predicts[task_name], all_inputs=all_inputs[task_name])
            all_metrics[res['task_name']] = res['result']/res['total']
            numbers+=res['result']#numbers.append(res['result'])
            total+=res['total']#total.append(res['total'])
            if args.local_rank in [0, -1]:
                print(res, flush=True)
        print(numbers/total)
        # all_metrics['z_mean'] = sum(numbers)/len(numbers)
        all_metrics['z_mean'] = numbers/total
        log2file(args, json.dumps(all_metrics, sort_keys=True))
        exit(0)

def main(args):
    # prepare model
    
    if args.tokenizer_name == 'qwen':
        tokenizer = AutoTokenizer.from_pretrained(
                                            args.model_path,
                                            max_length=CONFIG['max_len'],
                                            pad_token='<|endoftext|>',
                                            eos_token='<|endoftext|>',
                                            padding_side='right',
                                            trust_remote_code=True
                                        )
    else:
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
    if args.tokenizer_name in ['chatglm', 'internlm2']:
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 2
        tokenizer.unk_token_id = 0
    if args.tokenizer_name == 'bloom':
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 3
        tokenizer.unk_token_id = 0
    if args.tokenizer_name in ['phi', 'smollm2']:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print0('bos token id {} eos token id {} pad token id {}'.format(
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id
    ))
    tokenizer.add_special_tokens = False
    prompter = Prompter(args.template_name)
    CONFIG['tokenizer'] = tokenizer
    CONFIG['prompter'] = prompter

    if args.model_name == 'compressed':
        model = torch.load(os.path.join(args.model_path, 'compressed_model.bin'))
        if args.compressed_weights:
            model.load_state_dict(torch.load(
                os.path.join(args.compressed_weights, 'pytorch_model.bin'))
            )
    elif args.model_name == 'chatglm':
        model = AutoModel.from_pretrained(args.model_path,
                                            low_cpu_mem_usage=True,
                                            trust_remote_code=True,
                                            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16
                                            )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                            low_cpu_mem_usage=True,
                                            trust_remote_code=True,
                                            ignore_mismatched_sizes=True,
                                            attn_implementation=args.attn_implementation,
                                            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16
                                            )
        # try:
        #     from liger_kernel.transformers import apply_liger_kernel_to_llama, apply_liger_kernel_to_qwen2
        #     if model.config.model_type == 'llama':
        #         apply_liger_kernel_to_llama()
        #         print('using liger kernel for llama...')
        #     if model.config.model_type == 'qwen2':
        #         apply_liger_kernel_to_qwen2()
        #         print('using liger kernel for qwen2...')
        # except Exception as e:
        #     print(e)
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
    # if model.get_input_embeddings().weight.size(0) != len(tokenizer):
    #     model.resize_token_embeddings(len(tokenizer))
    
    total_params = sum(p.numel() for p in model.parameters())
    print0(f"Number of parameters: {total_params/1e9}B")
    # prepare data
    training_flags = args.instruct_tuning
    train_file_names = args.train_data
    val_file_names = args.val_data
    all_metrics = {}
    batched = True
    tok_func = functools.partial(generate_and_tokenize_prompt,
                                train_on_input=args.train_on_input,
                                batched=batched)
    val_file_names = glob.glob(args.val_data+args.val_files_pattern, recursive=True)
    train_file_names = glob.glob(args.train_data+args.train_files_pattern, recursive=True)

    # train each task separately
    if training_flags:
        train_dataset = load_dataset("json", 
                            data_files=train_file_names,
                            split='train',
                            streaming=args.streaming)
        if args.streaming:
            train_dataset = train_dataset.shuffle(seed=42, buffer_size=100000)\
                                                        .map(tok_func, batched=batched)
        else:
            train_dataset = train_dataset.shuffle(seed=42)\
                                                        .map(tok_func, batched=batched)

    eval_dataset = load_dataset("json", 
                            data_files=val_file_names,
                            split='train',
                            streaming=args.streaming,
                            )

    if args.streaming:
        eval_dataset = eval_dataset.shuffle(seed=42, buffer_size=100000)\
                                            .map(tok_func, batched=batched)
    else:
        eval_dataset = eval_dataset.shuffle(seed=42)\
                                .map(tok_func, batched=batched, batch_size=4096)

    if training_flags:
        curr_model = train(args, model, trainset=train_dataset, evalset=eval_dataset)
        curr_model.save_pretrained(
            os.path.join(args.output_dir, args.model_path.split('/')[-1]),
            safe_serialization=False
        )
        tokenizer.save_pretrained(
            os.path.join(args.output_dir, args.model_path.split('/')[-1])
        )
    else:
        curr_model = model
    if args.test:
        evaluate(args, curr_model, evalset=eval_dataset, all_metrics=all_metrics, training_flags=training_flags)
    
        if args.local_rank in [0, -1]:
            
            log2file(args, json.dumps(all_metrics, sort_keys=True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train/eval a model')
    parser.add_argument('-bsz', '--batch_size', type=int, \
                        default=CONFIG['batch_size'], help='global batch size')
    parser.add_argument('-m_bsz', '--micro_batch_size', type=int, \
                        default=CONFIG['micro_batch_size'], help='per gpu batch size')
    parser.add_argument('-e_bsz', '--eval_batch_size', type=int, \
                        default=CONFIG['eval_batch_size'], help='per gpu eval batch size')
    parser.add_argument('-output_dir', '--output_dir', type=str, \
                        default=CONFIG['output_dir'], help='output_dir')
    parser.add_argument('-acc_step',
                        '--accumulation_steps',
                        default=CONFIG['accumulation_steps'],
                        type=int,
                        required=False)
    parser.add_argument('-epochs', '--epochs', type=int, \
                        default=CONFIG['epochs'], help='training epochs')
    parser.add_argument('-max_steps', '--max_steps', type=int, \
                        default=CONFIG['max_steps'], help='training max_steps')
    parser.add_argument('-max_len', '--max_len', type=int, \
                        default=CONFIG['max_len'], help='training max_len')
    parser.add_argument('-save_steps', '--save_steps', type=int, \
                        default=CONFIG['save_steps'], help='save_steps')
    parser.add_argument('-eval_steps', '--eval_steps', type=int, \
                        default=CONFIG['eval_steps'], help='eval_steps')
    parser.add_argument('-max_new_tokens', '--max_new_tokens', type=int, \
                        default=800, help='max_new_tokens')
    parser.add_argument('-num_beams', '--num_beams', type=int, \
                        default=CONFIG['num_beams'], help='num_beams')
    parser.add_argument('-lr', '--learning_rate', type=float, \
                        default=CONFIG['learning_rate'], help='learning rate')
    parser.add_argument('-alpha', '--alpha', type=float, \
                        default=CONFIG['alpha'], help='weight of distillation loss')
    parser.add_argument('-temperature', '--temperature', type=float, \
                        default=CONFIG['temperature'], help='temperature for CE distillation loss')
    parser.add_argument('-v_data','--val_data', type=str, \
                        default=CONFIG['val_data'], help='the data used for evaluation')
    parser.add_argument('-t_data','--train_data', type=str, \
                        default=CONFIG['train_data'], help='the data used for instructing tuning')
    parser.add_argument('-p_data', '--pretrain_data', type=str, \
                        default=CONFIG['pretrain_data'], help='the data used for pretraining')
    parser.add_argument('--local_rank', default=-1, type=int,\
                        help='node rank for distributed training')
    parser.add_argument('--master_port', default="29501", type=str,\
                        help='master_port')
    parser.add_argument('--model_name', type=str, required=True,\
                        default=CONFIG['model_name'], help='the name of target llm model')
    parser.add_argument('--tokenizer_name', type=str, required=False,\
                        default='', help='the name of target llm tokenizer')
    parser.add_argument('--model_path', type=str, required=True,\
                        default=CONFIG['model_path'], help='the folder contains model weights')
    parser.add_argument('--student_model_path', type=str, required=False,\
                        default=CONFIG['student_model_path'], help='the folder contains student model weights')
    parser.add_argument('--compressed_weights', type=str, \
                        default="", help='the folder contains compressed model weights')
    parser.add_argument('--template_name', type=str, \
                        default='alpaca_short', help='instruct template')
    parser.add_argument('--loss_type', type=str, \
                        default=CONFIG['loss_type'], help='loss type')
    parser.add_argument('--deepspeed', type=str, \
                        default=CONFIG['deepspeed_config'], help='deepspeed config file path')
    parser.add_argument('-stage', '--stage', type=int, default=2, help='deepspeed stage')
    # CLUE /clue_pretrain_oscar_*.txt; xp3mt /**/*.jsonl
    parser.add_argument('--train_files_pattern', '-train_files_pattern', type=str, default='//*.jsonl')
    parser.add_argument('--val_files_pattern', '-val_files_pattern', type=str, default='//*.jsonl')
    parser.add_argument('-pt', '--pretrain', action='store_true',default= False)
    parser.add_argument('-do_eval', '--do_eval', action='store_true',default= False)
    parser.add_argument('-output', '--output_predict', action='store_true',default= False)
    parser.add_argument('-output_file', '--output_file',default="output.log")
    parser.add_argument('--attn_implementation', '-attn_impl', default='flash_attention_2')
    parser.add_argument('-num_gpus', '--num_gpus', type=int, default=8)
    parser.add_argument('-dora', '--dora', action='store_true',default= False)
    parser.add_argument('-moq', '--moq', action='store_true',default= False)
    parser.add_argument('-it', '--instruct_tuning', action='store_true',default=False)
    parser.add_argument('-fp16', '--fp16', action='store_true',default=False)
    parser.add_argument('-bf16', '--bf16', action='store_true',default=False)
    parser.add_argument('-offload', '--offload', action='store_true',default=False)
    parser.add_argument('--gen_config', type=str, \
                        default='default', help='generation config')
    parser.add_argument('-train_on_input', '--train_on_input', action='store_true',default=False)
    parser.add_argument('-distil', '--distil', action='store_true',default=False)
    parser.add_argument('-gradient_checkpointing', '--gradient_checkpointing', action='store_true',default= False)
    parser.add_argument('-multi_node', '--multi_node', action='store_true',default=False)
    parser.add_argument('-streaming', '--streaming', action='store_true',default=False)
    parser.add_argument('-encoded_data', '--encoded_data', action='store_true',default=False)
    parser.add_argument('-te', '--test', action='store_true', \
                        default=False, help='test the target model on downstream task')
    parser.add_argument('-vllm', '--eval_vllm', action='store_true', \
                        default=False, help='test the target model via vllm')
    parser.add_argument('-quant','--vllm_quant',type=str, help='quant mode in vllm', default='none')
    set_random_seed(19960301)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    args.output_file = os.path.join(args.output_dir, args.model_path.strip('/').split('/')[-1] + "-results.txt")
    if args.tokenizer_name == '':
        args.tokenizer_name = args.model_name
    args_dict = vars(args)
    for k, v in args_dict.items():
        CONFIG[k] = v
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if args.deepspeed:
        deepspeed.init_distributed("nccl")
    else:
        world_size = torch.cuda.device_count()
    CONFIG['world_size'] = world_size
    args.num_gpus = min(world_size, args.num_gpus)
    os.environ["MASTER_PORT"] = args.master_port
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    CONFIG['accumulation_steps'] = args.batch_size // args.micro_batch_size
    CONFIG['accumulation_steps'] = CONFIG['accumulation_steps'] // world_size
    args.accumulation_steps = CONFIG['accumulation_steps']
    pruned_path = os.path.join(args.model_path, 'pruned')
    if os.path.exists(pruned_path):
        args.model_path = pruned_path
    if args.local_rank in [0, -1]:
        print(CONFIG)
    datasets.config.IN_MEMORY_MAX_SIZE = 128 * 1024 * 1024
    if args.model_name in ['cpm']:
        args.eval_vllm = False
    if not args.instruct_tuning and args.eval_vllm:
        vllm_eval(args)
    else:
        main(args)
