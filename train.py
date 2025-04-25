# -- coding: utf-8 --**
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

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
# from prune_utils.composer_qwen2_ste import Qwen2ForL0Prune
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
    # print0(CONFIG['tokenizer'].batch_decode(input_ids))
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict_in_generate=True,
            min_new_tokens=1,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=CONFIG['tokenizer'].eos_token_id,
            pad_token_id=CONFIG['tokenizer'].pad_token_id
        )
    s = generation_output.sequences
    s = s[:, attention_mask.shape[-1]:]
    output = CONFIG['tokenizer'].batch_decode(s, skip_special_tokens=True)
    return output

def main(args):

    # prepare model
    if args.tokenizer_name in ['qwen']:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                                padding_side="right",
                                                pad_token='<|endoftext|>',
                                                eos_token='<|endoftext|>',
                                                truncation_side="right")
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
        )
    )
    tokenizer.add_special_tokens = False
    prompter = Prompter(args.template_name)
    CONFIG['tokenizer'] = tokenizer
    CONFIG['prompter'] = prompter
    model_dtype = torch.float16 if args.fp16 else torch.bfloat16
    if args.prune:
        if args.model_name in ['llama','deepseek','internlm2']:
            model = DeepSeekForL0Prune.from_pretrained(
                                                    args.model_path,
                                                    attn_implementation=args.attn_implementation,
                                                    torch_dtype=model_dtype)
        if args.instruct_tuning:
            from omegaconf import OmegaConf as om
            l0_config = om.load(args.l0_config)
            l0_config.model.l0_module.alpha = args.alpha
            l0_config.model.l0_module.sparsity_pool = args.sparsity_pool
            if args.max_steps > 0:
                l0_config.model.l0_module.lagrangian_warmup_steps = int(args.max_steps * args.lagrangian_warmup_ratio)
                l0_config.model.l0_module.max_prune_steps = int(args.max_steps * args.prune_step_ratio)
            if isinstance(model, PeftModel):
                model.base_model.model.init_l0_module(l0_config.model)
                model.base_model.model.model.l0_module.train()
                model.base_model.model.load_l0_parameters(args.model_path)
            else:
                model.init_l0_module(l0_config.model)
                model.model.l0_module.train()
                model.load_l0_parameters(args.model_path)
            model.train()
            if 'hidden' in l0_config.model.l0_module.pruning_modules:
                args.do_mid_distil = False
                CONFIG['do_mid_distil'] = False
                print0('setting `do_mid_distil` to False...')
            for pruning_module in l0_config.model.l0_module.pruning_modules:
                model.model.l0_module.masks[pruning_module].temperature = args.lag_temperature
        model = model.to(model_dtype)
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
    else:       
        model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                    low_cpu_mem_usage=True,
                                    trust_remote_code=True,
                                    ignore_mismatched_sizes=True,
                                    attn_implementation=args.attn_implementation,
                                    torch_dtype=model_dtype
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
                
    total_params = sum(p.numel() for p in model.parameters())
    print0(f"Number of parameters: {total_params/1e9}B")
    # prepare data
    training_flags = args.instruct_tuning or args.pretrain
    train_file_names = args.train_data
    val_file_names = args.val_data
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if not args.encoded_data: # load raw data and tokenize
        batched = True
        tok_func = functools.partial(generate_and_tokenize_prompt,
                                    train_on_input=args.train_on_input,
                                    batched=batched)
        if os.path.isdir(args.val_data):
            val_file_names = glob.glob(args.val_data+args.val_files_pattern, recursive=True)
            random.shuffle(val_file_names)
        
        eval_dataset = load_dataset("json", 
                                data_files=val_file_names,
                                split='train',
                                streaming=args.streaming,
                                )
        if args.streaming:
            eval_dataset = eval_dataset.shuffle(seed=args.seed, buffer_size=100000)\
                                                .map(
                                                    tok_func,
                                                    batched=batched,
                                                    )
            eval_dataset_features = list(list(eval_dataset.take(1))[0].keys())
        else:
            eval_dataset = eval_dataset.shuffle(seed=args.seed)\
                                        .map(
                                            tok_func,
                                            batched=batched,
                                            batch_size=4096,
                                            num_proc=os.cpu_count()
                                            )
            eval_dataset_features = list(eval_dataset.features.keys())
        if training_flags:
            if os.path.isdir(args.train_data):
                train_file_names = glob.glob(args.train_data+args.train_files_pattern, recursive=True)
                random.shuffle(train_file_names)
            train_dataset = load_dataset("json", 
                                data_files=train_file_names,
                                split='train',
                                streaming=args.streaming)

            if args.streaming:
                column_names = list(list(train_dataset.take(1))[0].keys())
                if 'weights' in column_names and not args.no_weight:
                    column_names.remove('weights')
                if 'input_ids' in column_names:
                    column_names.remove('input_ids')
                train_dataset = train_dataset.shuffle(seed=args.seed, buffer_size=100000)\
                                                .map(
                                                    tok_func,
                                                    batched=batched,
                                                    remove_columns=column_names
                                                    )
            else:
                column_names = list(train_dataset.features)
                train_dataset = train_dataset.shuffle(seed=args.seed)\
                                                .map(
                                                    tok_func,
                                                    batched=batched,
                                                    batch_size=4096,
                                                    num_proc=os.cpu_count(),
                                                    remove_columns=column_names
                                                    )
                if args.max_steps <= 0 and args.prune:
                    max_steps = len(train_dataset) * args.epochs //args.batch_size
                    model.model.l0_module.lagrangian_warmup_steps = int(max_steps * args.lagrangian_warmup_ratio)
                    model.model.l0_module.max_prune_steps = int(max_steps * args.prune_step_ratio)
            if args.pack:
                train_dataset = train_dataset.map(
                                                Concatenator(chunk_size=args.max_len),
                                                batched=batched,
                                                )
    else:
        eval_dataset = load_from_disk(args.val_data)
        train_dataset = load_from_disk(args.train_data)


    collate_fn = DataCollatorForSeq2Seq(CONFIG['tokenizer'],
                                        #pad_to_multiple_of=8,
                                        return_tensors="pt",
                                        padding='longest')
    

    if training_flags:
        training_ds_config = get_train_ds_config()
            offload=args.offload,
            stage=args.stage,
            enable_hybrid_engine=False,
            inference_tp_size=CONFIG['world_size'],
            max_out_tokens=2048
            )
        training_fsdp_config = get_train_fsdp_config()
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
            bf16=args.bf16,
            optim="adamw_torch",
            lr_scheduler_type=args.lr_scheduler,
            torch_compile=False,
            max_grad_norm=1.0,
            adam_beta1=0.9,
            adam_beta2=0.95,
            weight_decay=1e-4,
            logging_steps=10,
            remove_unused_columns=False,
            log_on_each_node=True,
            disable_tqdm=False,
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=args.eval_steps,
            load_best_model_at_end=False,
            save_steps = args.save_steps,
            save_total_limit=10,
            report_to='all',
            dataloader_num_workers=0,
            deepspeed=training_ds_config if args.deepspeed else None,
            fsdp=True if not args.deepspeed else False,
            fsdp_config=training_fsdp_config if not args.deepspeed else None,
            ddp_find_unused_parameters=False
        )

        if args.distil:
            print0('using Trainer: CustomTrainerForDistillation')
            trainer = CustomTrainerForDistillation(
                config=CONFIG,
                teacher_model_path=args.teacher_model_path,
                temperature=args.temperature,
                bild_topk=args.bild_topk,
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=CONFIG['tokenizer'],
                args=training_args,
                data_collator=collate_fn,
                callbacks=[ProcessBarCallback]
            )
        elif args.prune:
            from trainers.prune_trainer import (
                CustomTrainerForL0Prune, 
                CustomTrainerForDistillationL0Prune
            )
            if not args.teacher_model_path:
                print0('using Trainer: CustomTrainerForL0Prune')
                trainer = CustomTrainerForL0Prune(
                    config=CONFIG,
                    model=model,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    tokenizer=CONFIG['tokenizer'],
                    args=training_args,
                    data_collator=collate_fn,
                    callbacks=[ProcessBarCallback, PruneCallback]
                )
            else:
                print0('using Trainer: CustomTrainerForDistillationL0Prune')
                trainer = CustomTrainerForDistillationL0Prune(
                    config=CONFIG,
                    teacher_model_path=args.teacher_model_path,
                    temperature=args.temperature,
                    bild_topk=args.bild_topk,
                    model=model,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    tokenizer=CONFIG['tokenizer'],
                    args=training_args,
                    data_collator=collate_fn,
                    callbacks=[ProcessBarCallback, PruneCallback]
                )
        else:
            print0('using Trainer: CustomTrainerForSFT')
            trainer = CustomTrainerForSFT(
                config=CONFIG,
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=CONFIG['tokenizer'],
                args=training_args,
                data_collator=collate_fn,
                callbacks=[ProcessBarCallback]
            )
        if hasattr(trainer.accelerator, 'dataloader_config'):
            trainer.accelerator.dataloader_config.dispatch_batches=False
        else:
            trainer.accelerator.dispatch_batches=False
        if args.resume_from_checkpoint:
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        else:
            trainer.train()

    if args.test:
        CONFIG['tokenizer'].padding_side = "left"
        tokenizer.padding_side = "left"
        model.eval()
        if args.lora_weights or args.lora:
            model = model.merge_and_unload()
            model.train(False)
        if args.prune:
            model.convert_format()
            if args.save_pruned:
                output_dir = os.path.join(args.output_dir, 'pruned')
                tokenizer.save_pretrained(output_dir)
                model.save_pretrained(output_dir, safe_serialization=True)
                model.save_config(output_dir)
        if not args.deepspeed:
            if model.device == torch.device('cpu') and torch.cuda.device_count() > 0:
                model = model.to(torch.cuda.current_device())
        # not support bloom
        # model = BetterTransformer.transform(model)
        
        world_size = int(os.environ.get("WORLD_SIZE", 1))
 
        ds_config = {
            "replace_with_kernel_inject": True,
            "tensor_parallel": {
                "enabled": True,
                "tp_size": world_size
            },
        }
        if args.deepspeed and not training_flags:
            ds_engine = deepspeed.init_inference(
                                        model,
                                        dtype=torch.half,
                                        config=ds_config,
                                        )
            model = ds_engine.module
        def add_label_length(example):
            example['label_length'] = len(example['labels'])
            return example
        eval_dataset = eval_dataset.map(add_label_length)
        # 根据 'label_length' 进行排序
        eval_dataset = eval_dataset.sort('label_length', reverse=True)
        sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset,
                                    batch_size=args.eval_batch_size,
                                    collate_fn=eval_collate_fn,
                                    sampler=sampler,
                                    )
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
    parser.add_argument('-warmup_steps', '--warmup_steps', type=int, \
                        default=CONFIG['warmup_steps'], help='warmup_steps')
    parser.add_argument('-max_new_tokens', '--max_new_tokens', type=int, \
                        default=CONFIG['max_new_tokens'], help='max_new_tokens')
    parser.add_argument('-num_beams', '--num_beams', type=int, \
                        default=CONFIG['num_beams'], help='num_beams')
    parser.add_argument('-bild_topk', '--bild_topk', type=int, \
                        default=8, help='top k in bild distil loss')
    parser.add_argument('-lora_alpha', '--lora_alpha', type=int, \
                        default=32, help='lora alpha')
    parser.add_argument('-stage', '--stage', type=int, \
                        default=2, help='deepspeed stage')
    parser.add_argument('-lr', '--learning_rate', type=float, \
                        default=CONFIG['learning_rate'], help='learning rate')
    parser.add_argument('-lag_lr', '--lag_learning_rate', type=float, \
                        default=1.0, help='lag learning rate in l0 module')
    parser.add_argument('-lag_temp', '--lag_temperature', type=float, \
                        default=1.0/3.0, help='lag_temperature in l0 module')
    parser.add_argument('-lr_scheduler', '--lr_scheduler', type=str, \
                        default='cosine', help='lr_scheduler')
    parser.add_argument('-alpha', '--alpha', type=float, \
                        default=CONFIG['alpha'], help='alpha for l0 module')
    parser.add_argument('-sparsity_pool', '--sparsity_pool', type=float, \
                        default=CONFIG['sparsity_pool'], help='sparsity_pool for l0 module')
    parser.add_argument('-sparsity_noise', '--sparsity_noise', type=float, \
                        default=CONFIG['sparsity_noise'], help='max sparsity noise for l0 module')
    parser.add_argument('-lm_loss_weight', '--lm_loss_weight', type=float, \
                        default=CONFIG['lm_loss_weight'], help='weight of distillation loss')
    parser.add_argument('-loss_weights_2', '--loss_weights_2', type=float, \
                        default=CONFIG['loss_weights_2'], help='weight of distillation loss')
    parser.add_argument('-loss_weights_3', '--loss_weights_3', type=float, \
                        default=CONFIG['loss_weights_3'], help='weight of distillation loss')
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
    parser.add_argument('--num_relation_head', '-num_relation_head', default=CONFIG['num_relation_head'], type=int,\
                        help='number of relation heads in distillation')
    parser.add_argument('--master_port', default="29501", type=str,\
                        help='master_port')
    parser.add_argument('--model_name', type=str, required=True,\
                        default=CONFIG['model_name'], help='the name of target llm model')
    parser.add_argument('--model_path', type=str, required=True,\
                        default=CONFIG['model_path'], help='the folder contains model weights')
    parser.add_argument('--student_model_path', type=str, required=False,\
                        default=CONFIG['student_model_path'], help='the folder contains student model weights')
    parser.add_argument('--teacher_model_path', type=str, required=False,\
                        default='', help='the folder contains teacher model weights')
    parser.add_argument('--lora_weights', type=str, \
                        default="", help='the folder contains lora weights')
    parser.add_argument('--resume_from_checkpoint', type=str, \
                        default="", help='the folder contains checkpoint')
    parser.add_argument('--template_name', type=str, \
                        default='none', help='instruct template, see templates/*')
    parser.add_argument('--loss_type', type=str, \
                        default=CONFIG['loss_type'], help='loss type for finetune')
    parser.add_argument('-l0_config', '--l0_config', type=str, \
                        default='./prune_utils/config/baichuan2-3b.yaml', help='l0 prune config file')
    parser.add_argument('--attn_implementation', '-attn_impl', default='flash_attention_2')
    parser.add_argument('--deepspeed', type=str, \
                        default=None, help='use deepspeed or not')
    parser.add_argument('--train_files_pattern', '-train_files_pattern', type=str, default='//*.jsonl')
    parser.add_argument('--val_files_pattern', '-val_files_pattern', type=str, default='//*.jsonl')
    parser.add_argument('-pt', '--pretrain', action='store_true',default= False)
    parser.add_argument('-do_eval', '--do_eval', action='store_true',default= False)
    parser.add_argument('-output', '--output_predict', action='store_true',default= False)
    parser.add_argument('-output_file', '--output_file',default="output.log")
    parser.add_argument('-dora', '--dora', action='store_true',default= False)
    parser.add_argument('-gradient_checkpointing', '--gradient_checkpointing', action='store_true',default= False)
    parser.add_argument('-it', '--instruct_tuning', action='store_true',default=False)
    parser.add_argument('-te', '--test', action='store_true', default=False)
    parser.add_argument('-fp16', '--fp16', action='store_true',default=False)
    parser.add_argument('-bf16', '--bf16', action='store_true',default=False)
    parser.add_argument('-offload', '--offload', action='store_true',default=False)
    parser.add_argument('-train_on_input', '--train_on_input', action='store_true',default=False)
    parser.add_argument('-lora_from_ckpt', '--lora_from_checkpoint', action='store_true',default=False)
    parser.add_argument('-distil', '--distil', action='store_true',default=False)
    parser.add_argument('-prune', '--prune', action='store_true',default=False)
    parser.add_argument('-tokenizer_name', '--tokenizer_name', type=str,default='')
    parser.add_argument('-prune_step_ratio', '--prune_step_ratio', type=float,default=0.6)
    parser.add_argument('-lagrangian_warmup_ratio', '--lagrangian_warmup_ratio', type=float,default=0.6)
    parser.add_argument('-streaming', '--streaming', action='store_true',default=False)
    parser.add_argument('-pack', '--pack', action='store_true',default=False)
    parser.add_argument('-dev', '--dev', action='store_true',default=False)
    parser.add_argument('-no_weight', '--no_weight', action='store_true',default=False)
    parser.add_argument('-do_mid_distil', '--do_mid_distil', action='store_true',default=False)
    parser.add_argument('-save_pruned', '--save_pruned', action='store_true',default=False)
    parser.add_argument('-encoded_data', '--encoded_data', action='store_true',default=False)
    parser.add_argument('--seed', type=int,default=19960301)
    
    args = parser.parse_args()
    set_random_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    args.output_file = os.path.join(args.output_dir, "log.txt")
    if args.tokenizer_name == '':
        args.tokenizer_name = args.model_name
    args_dict = vars(args)
    for k, v in args_dict.items():
        CONFIG[k] = v
    device_map = "auto"
    local_rank = int(os.environ.get('LOCAL_RANK',0))
    args.local_rank = local_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # for mpirun
    if os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK'):
        os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] = os.environ.get('LOCAL_RANK')
    if os.environ.get('OMPI_COMM_WORLD_SIZE'):
        os.environ["OMPI_COMM_WORLD_SIZE"] = os.environ.get("WORLD_SIZE")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    CONFIG['world_size'] = world_size
    CONFIG['lora_model_loaded'] = False
    CONFIG['accumulation_steps'] = args.batch_size // args.micro_batch_size
    CONFIG['accumulation_steps'] = CONFIG['accumulation_steps'] // world_size
    args.accumulation_steps = CONFIG['accumulation_steps']
    if args.local_rank in [0, -1]:
        print0(CONFIG)
        log2file(args, CONFIG)
    datasets.config.IN_MEMORY_MAX_SIZE = 128 * 1024 * 1024
    main(args)
