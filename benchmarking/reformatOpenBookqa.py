import os
import json
import glob
import random
from datasets import load_dataset
from tqdm import tqdm
import sys
sys.path.append(os.getcwd())
# 添加父目录到模块搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompter import Prompter

datasets_dir='/mnt/data/JinQingyun/llm-workspace-sxh/benchmarking/datasets/'
task='siqa'
#train_files = '/mnt/data/JinQingyun/llm-workspace-sxh/benchmarking/datasets/openbookqa/train/train-additional.jsonl'#glob.glob(os.path.join(datasets_dir, task+'/train/*.json*'), recursive=True)
eval_files = '/mnt/data/JinQingyun/llm-workspace-sxh/benchmarking/datasets/mmlu/eval/mmlu.jsonl'#glob.glob(os.path.join(datasets_dir, task+'/eval/*.json*'), recursive=True)



prompter = Prompter(task)
# 转成 instruction, input, output, task_name 格式
train_data_file = os.path.join(datasets_dir, task, 'train/formated*.jsonl')

eval_data_file = os.path.join(datasets_dir, task, 'eval/formated*.jsonl')
trainset = load_dataset("json", data_files=train_data_file, split='train')
evalset = load_dataset("json", data_files=eval_data_file, split='train')
func = lambda item: {
    'instruction' : "",
    'input' : prompter.generate_prompt(item),
    'output' : item.get(prompter.template['label_key']),
    'task_name': task,
    'response_split': prompter.template['response_split']
}
remove_features = list(evalset.features.keys())
print('origin features in {} are'.format(task))
print(remove_features)
for feat in ['instruction', 'input', 'output', 'task_name', 'response_split']:
    if feat in remove_features:
        remove_features.remove(feat)

trainset = trainset.map(func, remove_columns=remove_features)
trainset.to_json(
    os.path.join(datasets_dir, task, f'train/input_train_{task}.jsonl'),
    batch_size=2048, num_proc=32, lines=True, force_ascii=False
)
evalset = evalset.map(func, remove_columns=remove_features)
evalset.to_json(
    os.path.join(datasets_dir, task, f'eval/input_eval_{task}.jsonl'),
    batch_size=2048, num_proc=32, lines=True, force_ascii=False
)
if task=='a':

    # 读取 lst 文件
    with open('/mnt/data/JinQingyun/llm-workspace-sxh/benchmarking/datasets/siqa/socialiqa-train-dev/dev-labels.lst', 'r') as f:
        answers_list = [line.strip() for line in f.readlines()]

    # 创建一个映射字典
    answer_mapping = {
        '1': 'answerA',
        '2': 'answerB',
        '3': 'answerC'
    }

    # 读取 jsonl 文件并添加 answer 字段
    new_jsonl_data = []
    with open('/mnt/data/JinQingyun/llm-workspace-sxh/benchmarking/datasets/siqa/socialiqa-train-dev/dev.jsonl', 'r') as f:
        for index, line in enumerate(f):
            json_obj = json.loads(line.strip())
            if index < len(answers_list):
                answer_index = answers_list[index]
                json_obj['answer'] = json_obj[answer_mapping[answer_index]]
            new_jsonl_data.append(json_obj)

    # 写入新的 jsonl 文件
    with open('/mnt/data/JinQingyun/llm-workspace-sxh/benchmarking/datasets/siqa/socialiqa-train-dev/formated-dev.jsonl', 'w') as f:
        for item in new_jsonl_data:
            f.write(json.dumps(item) + '\n')



if task == 'mmlu':
    def proc_func(item):
        return item['choices'][item['answer']]
        return None
    func = lambda item: {
        'task_name': 'mmlu',
        'input_question': item['question'],
        'options': item['choices'],
        'answer': proc_func(item)
    }
    remove_columns = ['question', 'subject', 'choices', 'answer']



    
    # process eval set
    evalset = load_dataset("json", 
                            data_files=eval_files,
                            split='train')
    evalset = evalset.map(
        func,
        remove_columns=remove_columns
    )
    evalset.to_json(os.path.join(datasets_dir, task, 'eval/formated_eval_{}.jsonl'.format(task)),
                        lines=True, force_ascii=False)
    
    ########################################
    prompter = Prompter(task)
    # 转成 instruction, input, output, task_name 格式
    #train_data_file = os.path.join(datasets_dir, task, 'train/formated*.jsonl')
    eval_data_file = os.path.join(datasets_dir, task, 'eval/formated*.jsonl')
    evalset = load_dataset("json", data_files=eval_data_file, split='train')
    func = lambda item: {
        'instruction' : "",
        'input' : prompter.generate_prompt(item),
        'output' : item.get(prompter.template['label_key']),
        'task_name': task,
        'response_split': prompter.template['response_split']
    }
    remove_features = list(evalset.features.keys())
    print('origin features in {} are'.format(task))
    print(remove_features)
    for feat in ['instruction', 'input', 'output', 'task_name', 'response_split']:
        if feat in remove_features:
            remove_features.remove(feat)

    
    evalset = evalset.map(func, remove_columns=remove_features)
    evalset.to_json(
        os.path.join(datasets_dir, task, f'eval/input_eval_{task}.jsonl'),
        batch_size=2048, num_proc=32, lines=True, force_ascii=False
    )

if task == 'openbookqa':
    def proc_func(item):
        return item['choices']['text'][ord(item['answerKey'])-ord('A')]
        return None
    func = lambda item: {
        'task_name': 'openbookqa',
        'input_question': item['question_stem'],
        'options': item['choices']['text'],
        'answer': proc_func(item)
    }
    remove_columns = ['id', 'question_stem', "choices",'answerKey']


    # process train set
    trainset = load_dataset("json", 
                            data_files=train_files,
                            split='train')
    trainset = trainset.map(
        func,
        remove_columns=remove_columns
    )
    trainset.to_json(os.path.join(datasets_dir, task, 'train/formated_train_{}.jsonl'.format(task)),
                        lines=True, force_ascii=False)
    
    # process eval set
    evalset = load_dataset("json", 
                            data_files=eval_files,
                            split='train')
    evalset = evalset.map(
        func,
        remove_columns=remove_columns
    )
    evalset.to_json(os.path.join(datasets_dir, task, 'eval/formated_eval_{}.jsonl'.format(task)),
                        lines=True, force_ascii=False)


    ########################################
    prompter = Prompter(task)
    # 转成 instruction, input, output, task_name 格式
    train_data_file = os.path.join(datasets_dir, task, 'train/formated*.jsonl')
    eval_data_file = os.path.join(datasets_dir, task, 'eval/formated*.jsonl')
    trainset = load_dataset("json", data_files=train_data_file, split='train')
    evalset = load_dataset("json", data_files=eval_data_file, split='train')
    func = lambda item: {
        'instruction' : "",
        'input' : prompter.generate_prompt(item),
        'output' : item.get(prompter.template['label_key']),
        'task_name': task,
        'response_split': prompter.template['response_split']
    }
    remove_features = list(trainset.features.keys())
    print('origin features in {} are'.format(task))
    print(remove_features)
    for feat in ['instruction', 'input', 'output', 'task_name', 'response_split']:
        if feat in remove_features:
            remove_features.remove(feat)

    trainset = trainset.map(func, remove_columns=remove_features)
    trainset.to_json(
        os.path.join(datasets_dir, task, f'train/input_train_{task}.jsonl'),
        batch_size=2048, num_proc=32, lines=True, force_ascii=False
    )
    evalset = evalset.map(func, remove_columns=remove_features)
    evalset.to_json(
        os.path.join(datasets_dir, task, f'eval/input_eval_{task}.jsonl'),
        batch_size=2048, num_proc=32, lines=True, force_ascii=False
    )