import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def bild_loss(logits_s, logits_t, top_k=8, temperature=3, student_led=False, norm=False):
    """
    Bi-directional Logits Difference loss.

    Args:
        logits_s (torch.Tensor): the student logits, shape (batch_size, seq_len, vocab_size).
        logits_t (torch.Tensor): the teacher logits, shape (batch_size, seq_len, vocab_size).
        top_k (int, optional): choose top-k logits for calculating loss, defaults to 8.
        temperature (int, optional): the temperature, defaults to 3.
        student_led (bool, optional): if true, calculate student-led logits difference loss (t-LD), else t-LD.
    """
    pair_num = top_k * (top_k-1) // 2

    if not student_led:
        # select top-k teacher logits & corresponding student logits
        if norm:
            with torch.no_grad():
                select_logits_t, select_pos = torch.topk(normalize(logits_t), k=top_k, dim=-1)
            select_logits_s = torch.gather(normalize(logits_s), 2, select_pos)
        else:
            with torch.no_grad():
                select_logits_t, select_pos = torch.topk(logits_t, k=top_k, dim=-1)
            select_logits_s = torch.gather(logits_s, 2, select_pos)
    else:
        # select top-k student logits & corresponding teacher logits
        if norm:
            select_logits_s, select_pos = torch.topk(normalize(logits_s), k=top_k, dim=-1)
            with torch.no_grad():
                select_logits_t = torch.gather(normalize(logits_t), 2, select_pos)
        else:
            select_logits_s, select_pos = torch.topk(logits_s, k=top_k, dim=-1)
            with torch.no_grad():
                select_logits_t = torch.gather(logits_t, 2, select_pos)

    scaled_logits_t = select_logits_t / temperature
    scaled_logits_s = select_logits_s / temperature

    # calculate logit difference
    def get_prob_diff(logits):
        b, n, v = logits.size()
        i, j = torch.triu_indices(v, v, offset=1)

        logits_diff = logits[..., i] - logits[..., j]

        return logits_diff

    logits_diff_t = get_prob_diff(scaled_logits_t)
    logits_diff_s = get_prob_diff(scaled_logits_s)

    logits_diff_t = F.softmax(logits_diff_t, dim=-1)

    loss = F.kl_div(F.log_softmax(logits_diff_s, dim=-1), logits_diff_t, reduction='none')

    loss = loss.sum(-1, keepdim=True)

    return loss

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


def bild_loss_new(logits_s, logits_t, top_k=8, temperature=3, norm=False):
    """
    Bi-directional Logits Difference loss.

    Args:
        logits_s (torch.Tensor): the student logits, shape (batch_size, seq_len, vocab_size).
        logits_t (torch.Tensor): the teacher logits, shape (batch_size, seq_len, vocab_size).
        top_k (int, optional): choose top-k logits for calculating loss, defaults to 8.
        temperature (int, optional): the temperature, defaults to 3.
        student_led (bool, optional): if true, calculate student-led logits difference loss (t-LD), else t-LD.
    """
    pair_num = top_k * (top_k-1) // 2
    mixed_logits = mix(logits_s, logits_t, 0.5)
    _, select_pos = torch.topk(mixed_logits, k=top_k, dim=-1)
    if norm:
        with torch.no_grad():
            select_logits_t = torch.gather(normalize(logits_t), 2, select_pos)
        select_logits_s = torch.gather(normalize(logits_s), 2, select_pos)
    else:
        with torch.no_grad():
            select_logits_t = torch.gather(logits_t, 2, select_pos)
        select_logits_s = torch.gather(logits_s, 2, select_pos)

    scaled_logits_t = select_logits_t / temperature
    scaled_logits_s = select_logits_s / temperature

    # calculate logit difference
    def get_prob_diff(logits):
        b, n, v = logits.size()
        i, j = torch.triu_indices(v, v, offset=1)

        logits_diff = logits[..., i] - logits[..., j]

        return logits_diff

    logits_diff_t = get_prob_diff(scaled_logits_t)
    logits_diff_s = get_prob_diff(scaled_logits_s)

    logits_diff_t_softmax = F.softmax(logits_diff_t, dim=-1)
    logits_diff_s_softmax = F.softmax(logits_diff_s, dim=-1)

    loss = F.kl_div(F.log_softmax(logits_diff_s, dim=-1), logits_diff_t_softmax, reduction='none')
    loss += F.kl_div(F.log_softmax(logits_diff_t, dim=-1), logits_diff_s_softmax, reduction='none')

    loss = loss.sum(-1, keepdim=True)

    return loss