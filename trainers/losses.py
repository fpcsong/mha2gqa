from config import *
from utils import print0
from distil_losses import top_kl_loss_func
# try:
#     from liger_kernel.transformers import LigerCrossEntropyLoss
#     ce_loss_func = LigerCrossEntropyLoss(reduction='none')
# except:
#     ce_loss_func = nn.CrossEntropyLoss(reduction='none')

ce_loss_func = nn.CrossEntropyLoss(reduction='none')

def mse_loss(logits_s, logits_t, temperature=1):
    num_pos_to_compute_loss = 1024
    beta_logits_t = logits_t
    beta_logits_s = logits_s

    with torch.no_grad():
        mask_pos = torch.argsort(beta_logits_t, dim=-1)
        select_pos = mask_pos[:,:, -num_pos_to_compute_loss:]
        select_logits_t = torch.gather(beta_logits_t, 2, select_pos)
        # select_logits_t = re_arange(select_logits_t)

    select_logits_s = torch.gather(beta_logits_s, 2, select_pos)
    loss = \
    F.mse_loss(select_logits_s.softmax(-1), select_logits_t.softmax(-1), reduction='none')
    return loss

def evaluate_confidence(logits, labels, temperature):
    """
    Evaluate the confidence for each position in the sequence.
    
    Args:
        logits: A tensor of shape (batchsize, seq_len, vocab_size) containing the output logits.
        labels: A tensor of shape (batchsize, seq_len) containing the ground truth labels.
        
    Returns:
        confidences: A tensor of shape (batchsize, seq_len) containing the inv confidence scores.
    """
    loss_mask = (labels > -1).long().float().to(logits.device)
    if temperature == 1.0:
        return loss_mask
    # Apply softmax to logits to get probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Gather the probabilities corresponding to the labels
    batch_size, seq_len = labels.shape
    confidences = probs[torch.arange(batch_size).unsqueeze(1), torch.arange(seq_len).unsqueeze(0), labels]
    confidences += (-1e6 * (1 - loss_mask))
    confidences = (confidences * temperature).softmax(-1) * loss_mask * loss_mask.sum(1).unsqueeze(1)  # sum is length
    confidences = confidences.clamp(min=1/temperature, max=temperature)
    inv_confidences = 1 / confidences * loss_mask
    return inv_confidences

def get_topk_lm_loss(logits, labels, k=1024):
    # full CE loss
    batch_size, seq_length, vocab_size = logits.shape
    if k < 0:
        select_logits = logits
        fake_labels = labels
        lm_loss = ce_loss_func(
            select_logits.reshape(-1, vocab_size), 
            fake_labels.reshape(-1)
        )
    else:
        logits_flat = logits.view(-1, vocab_size).detach().clone()
        with torch.no_grad():
            # 直接借助张量操作进行指数化或缩放
            indices = torch.where(labels.view(-1) > -1)[0]
            logits_flat[indices, labels.view(-1)[indices]] += 1e5
        # 对logits进行排序，只保留需要的前1024个位置，标签在第0位
        _, select_pos = logits_flat.topk(k, dim=-1)
        select_pos = select_pos.view(batch_size, seq_length, -1)
        select_logits = torch.gather(logits, 2, select_pos)

        fake_labels = torch.zeros_like(labels).long()
        lm_loss = ce_loss_func(
            select_logits.reshape(-1, k), 
            fake_labels.reshape(-1)
        )
    return lm_loss

def fused_loss(logits, labels, weights=None, temperature=3.0, length_norm=False, k=1024, incorrect_pos=None):
    
    batch_size, seq_length, vocab_size = logits.shape
    token_loss_weight = evaluate_confidence(logits, labels, temperature)
    loss_mask = (labels > -1).long().float().to(logits.device)
    if length_norm:
        token_loss_weight /= (loss_mask.sum(1, keepdim=True) + 1e-8)
        # token_loss_weight *= (loss_mask.sum(1, keepdim=True) + 1e-8) ** (1/5)
    if weights is not None:
        token_loss_weight *= weights.unsqueeze(1)

    # incorrect pos only
    if incorrect_pos is not None:
        token_loss_weight *= incorrect_pos

    token_loss_weight = token_loss_weight.reshape(-1)
    lm_loss = get_topk_lm_loss(logits, labels, k)
    # lm_loss = (lm_loss.reshape(-1) * token_loss_weight).sum() / token_loss_weight.sum()
    if length_norm:
        lm_loss = (lm_loss.reshape(-1) * token_loss_weight).sum() / batch_size
    else:
        lm_loss = (lm_loss.reshape(-1) * token_loss_weight).sum() / loss_mask.sum()
    return lm_loss

def normed_ce_loss(logits, labels, weights=None):
    
    batch_size, seq_length, vocab_size = logits.shape
    
    loss_mask = (labels > -1).long().float().to(logits.device)
    expected_number_of_tokens = loss_mask.sum()
    loss_mask /= (loss_mask.sum(1).unsqueeze(1) + 1e-8)
    if weights is not None:
        loss_mask *= weights.unsqueeze(1)
    loss_mask = loss_mask.reshape(-1)
    # expected_number_of_tokens = ((1+seq_length)/2) * batch_size
    loss_func = nn.CrossEntropyLoss(reduction='none')
    lm_loss = loss_func(logits.reshape(-1, vocab_size), 
                        labels.reshape(-1)
                        )
    lm_loss = (lm_loss.reshape(-1) * loss_mask).sum() / batch_size
    return lm_loss

def topk_normed_ce_loss(logits, labels, weights=None):
    
    num_pos_to_compute_loss = 1024

    batch_size, seq_length, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size).detach().clone()

    with torch.no_grad():
        # 直接借助张量操作进行指数化或缩放
        indices = torch.where(labels.view(-1) > -1)[0]
        logits_flat[indices, labels.view(-1)[indices]] += 1e5
    # 对logits进行排序，只保留需要的前1024个位置，标签在第0位
    _, select_pos = logits_flat.topk(num_pos_to_compute_loss, dim=-1)
    select_pos = select_pos.view(batch_size, seq_length, -1)

    select_logits = torch.gather(logits, 2, select_pos)

    loss_mask = (labels > -1).long().float().to(logits.device)
    loss_mask /= (loss_mask.sum(1, keepdim=True) + 1e-8)
    if weights is not None:
        loss_mask *= weights.unsqueeze(1)
    loss_mask = loss_mask.reshape(-1)
    fake_labels = torch.zeros_like(loss_mask)
    fake_labels = fake_labels.long()
    lm_loss = ce_loss_func(
        select_logits.reshape(-1, num_pos_to_compute_loss), 
        fake_labels.reshape(-1)
    )
    lm_loss = (lm_loss.reshape(-1) * loss_mask).sum() / batch_size
    return lm_loss


def re_arange(logits, ratio = 1):
    '''
    could 1/ratio works the same with temperature?
    '''
    with torch.no_grad():
        shape = logits.shape
        logits = logits.reshape(-1, logits.shape[-1])
        sorted_idxs = torch.argsort(logits, dim=-1)
        target_idxs_for_probs = torch.argsort(sorted_idxs, dim=-1)
        min_vals = logits.min(-1, keepdim=True)[0]
        logits_from_zero = logits - min_vals
        max_vals = logits_from_zero.max(-1, keepdim=True)[0]
        def func(x, a=1, b=1, c=0):
            return a * ((b*x)**4) + c
        x = []
        x_ends = torch.pow(max_vals, 1/4)
        for idx in range(x_ends.shape[0]):
            x.append(torch.linspace(0, x_ends[idx].item(), shape[-1]))
        x = torch.stack(x, 0)
        candicate_new_logits = func(x)
        candicate_new_logits = candicate_new_logits.to(logits.device)
        
        new_logits = torch.gather(candicate_new_logits, 1, target_idxs_for_probs)
        new_logits *= ratio
        new_logits += min_vals
        new_logits = new_logits.reshape(*shape)
        return new_logits

def compute_accuracy(logits, labels):
    valid_indices = (labels > -1)
    # 过滤有效位置的logits和labels
    valid_logits = logits[valid_indices]
    valid_labels = labels[valid_indices]
    
    # 计算预测结果
    predictions = torch.argmax(valid_logits, dim=-1)
    
    # 计算正确预测的数目
    correct_predictions = (predictions == valid_labels).sum().item()
    
    # 计算正确率
    accuracy = correct_predictions / valid_labels.size(0)
    
    return accuracy

import torch

def correct_logits(logits, labels):
    '''
    logits: 形状为 (batch, seq, vocab_size)，是模型输出的logits
    labels: 形状为 (batch, seq)，是真实标签
    '''
    batch_size, max_seq_len, vocab_size = logits.shape
    
    # Reshape
    reshape_logits = logits.view(-1, vocab_size).clone()
    reshape_labels = labels.view(-1)
    
    # 获取每个logit的最大值以及对应的索引
    max_logits, _ = reshape_logits.max(dim=1, keepdim=True)
    
    # 选取有效的样本
    valid_indices = (reshape_labels >= 0)
    
    # 获取预测值
    preds = torch.argmax(reshape_logits, dim=1)
    
    # 找到需要修正的索引
    need_correction = (preds != reshape_labels) & valid_indices

    # 调试信息
    # print("Logits before correction:", reshape_logits[need_correction, reshape_labels[need_correction]])
    
    # 安全进行logits校正
    correct_logits_val = max_logits[need_correction].squeeze() + 3
    indices_to_correct = reshape_labels[need_correction]
    reshape_logits[need_correction, indices_to_correct] = correct_logits_val
    
    # 调试信息
    # print("Logits after correction:", reshape_logits[need_correction, reshape_labels[need_correction]])
    
    # Reshape back to original shape
    corrected_logits = reshape_logits.view(batch_size, max_seq_len, vocab_size)
    
    # 再进行assert检查
    # assert torch.all(reshape_logits[valid_indices].max(dim=-1).indices == reshape_labels[valid_indices])
    
    return corrected_logits


def mix(logits_s, logits_t, alpha=0.8):
    with torch.no_grad():
        batch_size, max_seq_len, vocab_size = logits_s.shape
        t_min = logits_t.min(-1)[0].unsqueeze(-1).expand_as(logits_t)
        t_max = logits_t.max(-1)[0].unsqueeze(-1).expand_as(logits_t)
        s_min = logits_s.min(-1)[0].unsqueeze(-1).expand_as(logits_s)
        s_max = logits_s.max(-1)[0].unsqueeze(-1).expand_as(logits_s)
        scaled_logits_s = (logits_s - s_min ) / (s_max - s_min) # 0..1
        scaled_logits_s *= (t_max - t_min)
        scaled_logits_s += t_min
        mix_logits_t = logits_t * alpha + (1-alpha) * scaled_logits_s
    return mix_logits_t
