from .rouge import Rouge
import logging
from tqdm import tqdm
import json
import os
def print0(*message):
    """If distributed is initialized, print only on rank 0."""
    if int(os.environ.get('LOCAL_RANK', -1)) in [-1, 0]:
            print(*message, flush=True)

def evaluation_func(task, labels, predicts, all_inputs=None):

    if task in ['c3', 'clozet', 'cmrc2018', 'math23k', 'ocnli', 'sst2', 'wantwords','arc-c','arc-e','boolq','hellaswag','openbookqa','piqa','winogrande','mmlu','siqa']:
        total = len(labels) *1.0
        acc = 0
        for l,p in zip(labels, predicts):
            if l.strip() == p.strip():
                acc += 1
        return {'task_name': task, 'metric': 'accuracy', 'result': acc,'total':total,'acc':acc/total}
    if task in ['qwen']:
        total = len(labels) *1.0
        acc = 0
        positive=0
        recall=0
        for l,p in zip(labels, predicts):
            if l.strip() == p.strip():
                acc += 1
            if l.strip() =='是':
                positive+=1
                if l.strip() == p.strip():
                    recall+=1
        return {'task_name': task, 'metric': 'recall', 'result': recall/positive}
    if task in ['cluener']:
        tp, fp, fn = 0, 0, 0
        for gold, pre in zip(labels, predicts):
            gold_kv = []
            for kvs in gold.split(';'):
                # k, vs = kvs.split(':')
                k = kvs.split(':')[0]
                vs = ''.join(kvs.split(':')[1:])
                for v in vs.split(','):
                    gold_kv.append(k+v)
            pre_kv = []
            for all_kvs in pre.split(';'):
                kvs = all_kvs.split(':')
                if len(kvs) > 1:
                    vs = kvs[1].split(',')
                    for v in vs:
                        pre_kv.append(kvs[0]+v)
            for item in pre_kv:
                if item in gold_kv:
                    tp += 1
                else:
                    fp += 1
            for item in gold_kv:
                if item not in pre_kv:
                    fn += 1
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f = 2 * p * r / (p + r + 1e-8)
        return {'task_name': task, 'metric': 'f1', 'result': f}

    if task == 'lcsts':
        m = Rouge(variants=[1,2,'L'], multiref='best')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('./tokenizer')
        preds = []
        y = []
        for l, p in zip(labels, predicts):
            y = tokenizer.tokenize(l)
            preds = tokenizer.tokenize(p)
            m.update(([preds], [[y]]))
        res = m.compute()
        rouge_1_F = res['Rouge-1-F']
        rouge_2_F = res['Rouge-2-F']
        rouge_L_F = res['Rouge-L-F']
        # result = (rouge_1_F + rouge_2_F + rouge_L_F) / 3
        result = rouge_1_F

        return {'task_name': 'lcsts', 'metric' : 'rouge', 'result': result}
    # if task == 'lcsts':
    #     from transformers import AutoTokenizer
    #     tokenizer = AutoTokenizer.from_pretrained('./tokenizer')
        
    #     rouge = Rouge()
    #     from transformers import AutoTokenizer
    #     tokenizer = AutoTokenizer.from_pretrained('./tokenizer')
    #     y = [" ".join(tokenizer.tokenize(l)) for l in labels]
    #     preds = [" ".join(tokenizer.tokenize(p)) for p in predicts]
    #     scores = rouge.get_scores(preds, y)
    #     rouge_1_F = [item['rouge-1']['f'] for item in scores]
    #     rouge_1_F = sum(rouge_1_F) / len(rouge_1_F)
    #     result = rouge_1_F
    #     return {'task_name': 'lcsts', 'metric' : 'rouge', 'result': result}
    
    def check_filed(dict1, dict2):
        '''for dict that only contains str values'''
        def isSubsequence(s: str, t: str) -> bool:
            dp = [[0] * (len(t)+1) for _ in range(len(s)+1)]
            for i in range(1, len(s)+1):
                for j in range(1, len(t)+1):
                    if s[i-1] == t[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = dp[i][j-1]
            if dp[-1][-1] == len(s):
                return True
            return False
        if dict1.keys() != dict2.keys():
            return False
        for k in dict1:
            if isSubsequence(dict1.get(k), dict2.get(k)) or isSubsequence(dict2.get(k), dict1.get(k)):
                continue
            return False
        return True
    if task in ['entitycombine']:
        tp, fp, fn = 0, 0, 0
        for gold, pre, inp in zip(labels, predicts, all_inputs):
            gold_kv = []
            pre_kv = []
            gjson = json.loads(gold)
            keys=gjson.keys()# 联系人，日程
            gt_items = []
            pred_items = []
            for key in keys:
                item = gjson[key]
                if not item:
                    continue
                for sub_item in item:
                    if key == '联系人':
                        if not sub_item.get('联系人名') or not sub_item.get('电话号码'):
                            continue
                        if sub_item['工作单位'] in sub_item['联系人名']:
                            sub_item['工作单位'] = ''
                        if sub_item['职位'] in sub_item['联系人名']:
                            sub_item['职位'] = ''
                    if key == '日程':
                        if not sub_item.get('待办事件') or not sub_item.get('时间点'):
                            continue
                    gt_items.append(sub_item)
            try:
                pjson = json.loads(pre)
            except:
                num = len(gt_items)
                fp += len(gt_items)
                continue

            keys=pjson.keys()# 联系人，日程
            for key in keys:
                item = pjson[key]
                if not item:
                    continue
                for sub_item in item:
                    if key == '联系人':
                        if not sub_item.get('联系人名') or not sub_item.get('电话号码'):
                            continue
                        if sub_item['工作单位'] in sub_item['联系人名']:
                            sub_item['工作单位'] = ''
                        if sub_item['职位'] in sub_item['联系人名']:
                            sub_item['职位'] = ''
                    if key == '日程':
                        if not sub_item.get('待办事件') or not sub_item.get('时间点'):
                            continue
                    pred_items.append(sub_item)
            err_g_items = []
            err_p_items = []
            for p_item in pred_items:
                hit = False
                for g_item in gt_items:
                    if check_filed(p_item, g_item):
                        tp += 1
                        hit = True
                        break
                if not hit:
                    fp += 1
                    err_p_items.append(p_item)
            for g_item in gt_items:
                hit = False
                for p_item in pred_items:
                    if check_filed(g_item, p_item):
                        hit = True
                        break
                if not hit:
                    fn += 1
                    err_g_items.append(g_item)
            # if err_g_items or err_p_items:
            #     print0(inp.split('下面给出这段文字:')[1])
            #     print0('label:')
            #     print0(err_g_items)
            #     print0('pred:')
            #     print0(err_p_items)
            #     print0('tp-fp-fn:')
            #     print0('*' * 20)

        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f = 2 * p * r / (p + r + 1e-8)
        print0(tp, fp, fn)
        return {'task_name': task, 'metric': 'f1', 'result': f}