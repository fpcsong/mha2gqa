# -- coding: utf-8 --**
from config import *

def tokenize_it(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = CONFIG['tokenizer'](
        prompt,
        truncation=True,
        max_length=CONFIG['max_len'],
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != CONFIG['tokenizer'].eos_token_id
        and len(result["input_ids"]) < CONFIG['max_len']
        and add_eos_token
    ):
        result["input_ids"].append(CONFIG['tokenizer'].eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result
def tokenize_pt(example, add_eos_token=True):
    # there's probably a way to do this with the CONFIG['tokenizer'] settings
    # but again, gotta move fast
    result = CONFIG['tokenizer'](
        ''.join(example['clean_doc']),
        truncation=True,
        max_length=CONFIG['max_len'],
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != CONFIG['tokenizer'].eos_token_id
        and len(result["input_ids"]) < CONFIG['max_len']
        and add_eos_token
    ):
        result["input_ids"].append(CONFIG['tokenizer'].eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_points, train_on_input=False, batched=False):
    batch_size = 1
    if not batched:
        for k in data_points.keys():
            data_points[k] = [data_points[k]]
    else:
        for k in data_points.keys():
            batch_size = max(batch_size, len(data_points.get(k)))
    
    samples = [dict([(k, data_points[k][i]) \
                     for k in data_points.keys()]) \
                        for i in range(batch_size)]
    # for pretraining
    if 'input_ids' in data_points:
        ret = {}
        ret['input_ids'] = [item['input_ids'][:CONFIG['max_len']] for item in samples]
        ret['labels'] = ret['input_ids'].copy()
        ret['attention_mask'] = [[1] * len(item) for item in ret['input_ids']]
        ret['weights'] = [1] * batch_size
        return ret
    
    user_prompts = [CONFIG['prompter'].generate_prompt(sample) for sample in samples]
    weights = [1.0] * len(samples)
    if 'weights' in samples[0]:
        weights = [sample['weights'] for sample in samples]
    tokenized_user_prompt = CONFIG['tokenizer'](
        user_prompts,
        truncation=True,
        max_length=CONFIG['max_len'],
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
    )
    if not train_on_input:
        tokenized_user_prompt['labels'] = [[-100] * len(item) \
                                           for item in tokenized_user_prompt['input_ids']]
    else:
        tokenized_user_prompt['labels'] = tokenized_user_prompt['input_ids'].copy()

    labels = data_points.get(CONFIG['prompter'].template['label_key'])
    if labels is None:
        return tokenized_user_prompt

    tokenized_label = CONFIG['tokenizer'](
        labels,
        truncation=True,
        max_length=CONFIG['max_len'],
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
    )
    # add eos token
    for idx, item in enumerate(tokenized_label['input_ids']):
        if item[-1] != CONFIG['tokenizer'].eos_token_id and len(item) < CONFIG['max_len']:
            tokenized_label['input_ids'][idx].append(CONFIG['tokenizer'].eos_token_id)
            tokenized_label['attention_mask'][idx].append(1)

    tokenized_label['labels'] = tokenized_label['input_ids'].copy()

    prompt_input_ids = tokenized_user_prompt['input_ids']
    label_input_ids = tokenized_label['input_ids']

    prompt_attention_mask = tokenized_user_prompt['attention_mask']
    label_attention_mask = tokenized_label['attention_mask']

    prompt_labels = tokenized_user_prompt['labels']
    label_labels = tokenized_label['labels']

    ret = {}
    ret['input_ids'] = [prompt_input_ids[k] + label_input_ids[k] for k in range(batch_size)]
    ret['attention_mask'] = [prompt_attention_mask[k] + label_attention_mask[k] for k in range(batch_size)] 
    ret['labels'] = [prompt_labels[k] + label_labels[k] for k in range(batch_size)]
    ret['weights'] = weights
    return ret