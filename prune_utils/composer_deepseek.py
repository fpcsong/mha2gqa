# Copyright 2023 Baichuan Inc. All Rights Reserved.

# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.




# modified from https://github.com/princeton-nlp/LLM-Shearing/blob/main/llmshearing/models/composer_llama.py
# 2024-01, songxiaohui@oppo.com

from .l0_module import L0Module
from .modeling_flash_attention_utils import *
from .modeling_flash_attention_utils import _flash_attention_forward
import inspect
import math
from typing import List, Optional, Tuple, Union
from threading import Thread
from tqdm import tqdm
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation.utils import GenerationConfig
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
    ContextManagers
)
from transformers.pytorch_utils import (
                                        find_pruneable_heads_and_indices
                                        )
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
)
from transformers import LlamaConfig
from glob import glob
from omegaconf import OmegaConf as om
from collections import OrderedDict
from safetensors.torch import load_file

import os
from contextlib import contextmanager
logger = logging.get_logger(__name__)

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

def print0(*message):
    """If distributed is initialized, print only on rank 0."""
    if int(os.environ.get('LOCAL_RANK', -1)) in [-1, 0]:
            print(*message, flush=True)


def turn_head_z(head_z, head_layer_z):
    head_z = head_z.squeeze().clone()
    if head_layer_z is not None:
        head_z *= head_layer_z
    to_prune_heads = torch.where(head_z == 0)[0].view(-1).tolist()
    return to_prune_heads

def turn_mlp_z(intermediate_z, mlp_z, head_layer_z):
    intermediate_z_layer = intermediate_z.squeeze().clone()
    if mlp_z is not None:
        intermediate_z_layer *= mlp_z
    if head_layer_z is not None:
        intermediate_z_layer *= head_layer_z
    keep_intermediate_dims = torch.where(intermediate_z_layer > 0)[0].tolist()
    return keep_intermediate_dims

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, hidden_z=None):
        '''
        if hidden_z is not None:
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            compressed_input = torch.index_select(hidden_states, dim=-1, index=remaining_index)
        else:
            compressed_input = hidden_states
        variance = compressed_input.to(torch.float32).pow(2).mean(-1, keepdim=True)
        '''
        if hidden_z is not None:
            variance = hidden_states.mul(hidden_z).pow(2).sum(-1, keepdim=True) / hidden_z.sum()
        else:
            variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        
        output = self.weight * hidden_states
        if hidden_z is not None:
            output = output.mul(hidden_z)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            output = output.to(self.weight.dtype)
        return output
    def prune_params(self, hidden_z):
        remaining_index = torch.where(~hidden_z.eq(0))[0]
        self.weight = torch.nn.Parameter(self.weight.data.mul(hidden_z.squeeze())[remaining_index])

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`torch.nn.Linear`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `torch.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def prune_params(self, zs_layer):
        intermediate_size = self.down_proj.in_features
        intermediate_z = zs_layer.get("intermediate_z", None)
        mlp_z = zs_layer.get("mlp_z", None)
        hidden_z = zs_layer.get("hidden_z", None)
        head_layer_z= zs_layer.get("head_layer_z", None)

        dtype = next(self.up_proj.parameters()).dtype
        # update params #
        if intermediate_z is not None:
            self.up_proj.weight.data = self.up_proj.weight.data.transpose(0, 1).mul(intermediate_z.squeeze(0)).transpose(0, 1)
        if mlp_z is not None:
            self.down_proj.weight.data = self.down_proj.weight.data.transpose(0, 1).mul(mlp_z).transpose(0, 1)
        if head_layer_z is not None:
            self.down_proj.weight.data = self.down_proj.weight.data.transpose(0, 1).mul(head_layer_z).transpose(0, 1)
        if hidden_z is not None:
            self.down_proj.weight.data = self.down_proj.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, 1) 
        #################
        if hidden_z is not None:
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            dtype = next(self.up_proj.parameters()).dtype
            print0(f"    FFN hidden dim: {len(hidden_z)} -> {len(remaining_index)}")
            self.up_proj = prune_linear_layer(self.up_proj, remaining_index, dim=1)
            self.gate_proj = prune_linear_layer(self.gate_proj, remaining_index, dim=1)
            self.down_proj = prune_linear_layer(self.down_proj, remaining_index, dim=0)
            self.up_proj = self.up_proj.to(dtype)
            self.gate_proj = self.gate_proj.to(dtype)
            self.down_proj = self.down_proj.to(dtype)
        if intermediate_z is not None:
            keep_dim = turn_mlp_z(intermediate_z, mlp_z, head_layer_z)
            # print0(keep_dim, intermediate_z.sum())
            device = self.up_proj.weight.device
            if len(keep_dim) == self.up_proj.weight.shape[0]:
                print0(f"1    FFN intermediate dim: {intermediate_size} -> {len(keep_dim)}")
                return 
            
            if len(keep_dim) == 0:
                self.up_proj = None; self.down_proj = None; self.gate_proj = None
            else:
                keep_dim_index = torch.tensor(keep_dim).long().to(device)
                dtype = next(self.up_proj.parameters()).dtype
                self.up_proj = prune_linear_layer(self.up_proj, keep_dim_index, dim=0)
                self.gate_proj = prune_linear_layer(self.gate_proj, keep_dim_index, dim=0)
                self.down_proj = prune_linear_layer(self.down_proj, keep_dim_index, dim=1)
                self.up_proj = self.up_proj.to(dtype)
                self.gate_proj = self.gate_proj.to(dtype)
                self.down_proj = self.down_proj.to(dtype)
            print0(f"2    FFN intermediate dim: {intermediate_size} -> {len(keep_dim)}")
    def forward(self, x, intermediate_z=None, mlp_z=None, hidden_z=None, head_layer_z=None):
        if self.up_proj is None:
            return None
        gate = self.act_fn(self.gate_proj(x))
        up_v = self.up_proj(x)

        if intermediate_z is not None:
            up_v *= intermediate_z

        down_v = self.down_proj(gate * up_v)
        if mlp_z is not None:
            down_v = down_v * mlp_z

        if hidden_z is not None:
            down_v = down_v * hidden_z

        if head_layer_z is not None:
            down_v = down_v * head_layer_z
        return down_v

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.pruned_heads = set()
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.kv_layer = None
        self._init_rope()
    def _init_rope(self, device=None):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
                device=device
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    device=device
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    device=device
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    def convert_mha_to_gqa(self, mha_kv_weight, num_heads, group_size=4):
        """
        转换多头注意力的KV部分为分组查询注意力（GQA）。
        
        参数:
            mha_kv_weight (torch.Tensor): MHA模块中的KV部分的权重。形状为 (num_heads * head_dim, embed_dim)
            num_heads (int): 原始MHA中的头数。
            group_size (int): 每组的头数，默认为4。
            
        返回:
            torch.Tensor: 新的GQA/分组的KV权重。形状为 (new_num_heads * head_dim, embed_dim)。
        """
        head_dim = mha_kv_weight.size(0) // num_heads
        grouped_heads = num_heads // group_size
        new_kv_weight = torch.zeros(grouped_heads * head_dim, mha_kv_weight.size(1)).to(mha_kv_weight.device)
        
        for group in range(grouped_heads):
            # 对每个组进行平均
            start_head = group * group_size
            end_head = (group + 1) * group_size
            heads_weight = []
            
            for head in range(start_head, end_head):
                start_index = head * head_dim
                end_index = (head + 1) * head_dim
                heads_weight.append(mha_kv_weight[start_index:end_index, :])
            # 计算这一组的权重和，因为此前已经乘以权重了
            heads_weight = torch.stack(heads_weight, dim=0)
            mean_weight = torch.mean(heads_weight, dim=0)
            new_kv_weight[group * head_dim:(group + 1) * head_dim, :] = mean_weight
        
        return new_kv_weight
    def prune_params(self, zs_layer):
        head_z = None; head_layer_z = None; hidden_z = None; qk_head_dim_z = None; vo_head_dim_z = None; kv_head_z=None
        if "head_z" in zs_layer:
            head_z = zs_layer["head_z"].squeeze()

        if "kv_head_z" in zs_layer:
            kv_head_z = zs_layer["kv_head_z"].squeeze()

        if "head_layer_z" in zs_layer:
            head_layer_z = zs_layer["head_layer_z"].squeeze()
        
        if "hidden_z" in zs_layer:
            hidden_z = zs_layer["hidden_z"].squeeze()
        
        if "qk_head_dim_z" in zs_layer:
            qk_head_dim_z = zs_layer["qk_head_dim_z"].squeeze() # qk_head_dim is the same as hidden_z
            vo_head_dim_z = zs_layer["vo_head_dim_z"].squeeze() # vo_head_dim is the same as hidden_z
        # update params #
        if head_z is not None:
            head_z_for_update = torch.repeat_interleave(head_z, self.head_dim)
            # self.v_proj.weight.data = self.v_proj.weight.data.transpose(0, 1).mul(head_z_for_update).transpose(0, 1)
            self.q_proj.weight.data = self.q_proj.weight.data.transpose(0, 1).mul(head_z_for_update).transpose(0, 1)
        if kv_head_z is not None:
            if head_z is not None:
                kv_head_z = kv_head_z * head_z.reshape(kv_head_z.shape)
            
            num_kv_head_kept = (kv_head_z > 0).long().float().sum(-1).mean()
            normed_kv_head_z = kv_head_z / num_kv_head_kept
            kv_head_z_for_update = torch.repeat_interleave(normed_kv_head_z.reshape(-1), self.head_dim)
            # print0(kv_head_z_for_update.mean(), kv_head_z_for_update.max(), kv_head_z_for_update.min())
            # kv_head_z_for_update = torch.ones_like(kv_head_z_for_update) / kv_head_z.shape[-1]

            self.k_proj.weight.data = self.k_proj.weight.data.transpose(0, 1).mul(kv_head_z_for_update).transpose(0, 1)
            self.v_proj.weight.data = self.v_proj.weight.data.transpose(0, 1).mul(kv_head_z_for_update).transpose(0, 1)
        if head_layer_z is not None:
            self.o_proj.weight.data = self.o_proj.weight.data.transpose(0, 1).mul(head_layer_z).transpose(0, 1)
        if hidden_z is not None:
            self.o_proj.weight.data = self.o_proj.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, 1)
        if qk_head_dim_z is not None:
            self.q_proj.weight.data = self.q_proj.weight.data.transpose(0, 1).mul(qk_head_dim_z).transpose(0, 1)
            self.v_proj.weight.data = self.v_proj.weight.data.transpose(0, 1).mul(vo_head_dim_z).transpose(0, 1)
        #################

        dtype = next(self.o_proj.parameters()).dtype
        if hidden_z is not None:
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            print0(f"    Head hidden: {len(hidden_z)} -> {len(remaining_index)}") 
            self.k_proj = prune_linear_layer(self.k_proj, remaining_index, dim=1)
            self.q_proj= prune_linear_layer(self.q_proj, remaining_index, dim=1)
            self.v_proj = prune_linear_layer(self.v_proj, remaining_index, dim=1)
            self.o_proj = prune_linear_layer(self.o_proj, remaining_index)
            self.q_proj = self.q_proj.to(dtype)
            self.k_proj = self.k_proj.to(dtype)
            self.v_proj = self.v_proj.to(dtype)
            self.o_proj = self.o_proj.to(dtype)

        # query head
        if head_z is not None:
            to_prune_heads = turn_head_z(head_z, head_layer_z)
            len_to_prune_heads = len(to_prune_heads)
            if len_to_prune_heads == 0 and qk_head_dim_z is None:
                print0(f"1    Heads: {self.num_heads} -> {self.num_heads}")
                return

            heads, index = find_pruneable_heads_and_indices(
                to_prune_heads, self.num_heads, self.head_dim, self.pruned_heads
            )
            # Update hyper params and store pruned heads
            print0(f"2    Heads: {self.num_heads} -> {self.num_heads - len(heads)}")
            self.num_heads = self.num_heads - len(heads)
            self.pruned_heads = self.pruned_heads.union(heads)
            
            if kv_head_z is None:
                self.num_key_value_heads = self.num_key_value_heads - len(heads)
            # Prune linear layers
            # setting layers to be None if all the heads are pruned
            if self.num_heads == 0:
                self.q_proj = None
                self.k_proj = None
                self.v_proj = None
                self.o_proj = None
                return
            else:
                qk_index = index; vo_index = index
                if qk_head_dim_z is not None:
                    remaining_qk_index = torch.where(~qk_head_dim_z.eq(0))[0]
                    remaining_vo_index = torch.where(~vo_head_dim_z.eq(0))[0]
                    import numpy as np
                    qk_index = torch.from_numpy(
                        np.intersect1d(index.detach().cpu().numpy(), remaining_qk_index.detach().cpu().numpy())
                        ).to(index.device).to(index.dtype)
                    vo_index = torch.from_numpy(
                        np.intersect1d(index.detach().cpu().numpy(), remaining_vo_index.detach().cpu().numpy())
                        ).to(index.device).to(index.dtype)
                    print0(f"    QKVO dims: {len(hidden_z)} -> {len(qk_index)}")
                    
                    self.head_dim = len(qk_index) // self.num_heads
                    self._init_rope(device=self.q_proj.weight.device)

                self.q_proj= prune_linear_layer(self.q_proj, qk_index)
                self.o_proj = prune_linear_layer(self.o_proj, vo_index, dim=1)
        if self.num_key_value_groups == 1 and kv_head_z is None:# MHA
            self.k_proj = prune_linear_layer(self.k_proj, qk_index)
            self.v_proj = prune_linear_layer(self.v_proj, vo_index)
        elif kv_head_z is not None and self.q_proj is not None:

            self.k_proj.weight.data = self.kv_layer[0].weight.data
            self.v_proj.weight.data = self.kv_layer[1].weight.data
            self.kv_layer = None

            # self.k_proj.weight.data = self.convert_mha_to_gqa(self.k_proj.weight.data, self.num_heads, kv_head_z.shape[-1])
            # self.v_proj.weight.data = self.convert_mha_to_gqa(self.v_proj.weight.data, self.num_heads, kv_head_z.shape[-1])

            print0(f"    num KV heads: {self.num_key_value_heads} -> {kv_head_z.shape[0]}")
            self.num_key_value_heads = kv_head_z.shape[0]
            self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = self.q_proj.to(dtype)
        self.k_proj = self.k_proj.to(dtype)
        self.v_proj = self.v_proj.to(dtype)
        self.o_proj = self.o_proj.to(dtype)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        head_z=None,
        head_layer_z=None,
        hidden_z=None,
        qk_head_dim_z=None,
        vo_head_dim_z=None,
        kv_head_z=None, # n_kv_head, len_group
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if qk_head_dim_z is not None:
            query_states = query_states.mul(qk_head_dim_z)
            query_states = query_states.to(key_states.dtype)
        if vo_head_dim_z is not None:
            value_states = value_states.mul(vo_head_dim_z)
            value_states = value_states.to(key_states.dtype)

        if kv_head_z is not None: # only work for MHA model
            if head_z is not None:
                kv_head_z = kv_head_z * head_z.reshape(kv_head_z.shape)
            '''
            # 0 ~ 1
            ratio = (kv_head_z.shape[-1] - kv_head_z.sum(-1)) / (kv_head_z.shape[-1] - 1)
            ratio = ratio.clamp(0, 1)
            ratio = ratio.unsqueeze(-1).expand_as(kv_head_z)
            # 1 ~ 0
            ratio = 1 - ratio
            kv_head_z_weights = torch.repeat_interleave(ratio.reshape(-1), self.head_dim)
            '''
            kv_head_z_weights = torch.repeat_interleave(kv_head_z.reshape(-1), self.head_dim)

            selected_key_states = self.kv_layer[0](hidden_states)
            selected_key_states = selected_key_states.reshape(bsz, q_len, kv_head_z.shape[0], -1)
            selected_key_states = torch.repeat_interleave(selected_key_states, kv_head_z.shape[-1], 2)
            selected_key_states = selected_key_states.reshape(bsz, q_len, -1)
            selected_value_states = self.kv_layer[1](hidden_states)
            selected_value_states = selected_value_states.reshape(bsz, q_len, kv_head_z.shape[0], -1)
            selected_value_states = torch.repeat_interleave(selected_value_states, kv_head_z.shape[-1], 2)
            selected_value_states = selected_value_states.reshape(bsz, q_len, -1)

            key_states = key_states * kv_head_z_weights + selected_key_states * (1 - kv_head_z_weights)
            value_states = value_states * kv_head_z_weights + selected_value_states * (1 - kv_head_z_weights)

            key_states = key_states.to(query_states.dtype)
            value_states = value_states.to(query_states.dtype)
        # batch, num_heads, seq_len, head_dim
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        # batch, num_key_value_heads * n_rep, slen, head_dim
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        if head_z is not None:
            attn_output *= head_z.unsqueeze(-1)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if head_layer_z is not None:
            attn_output *=  head_layer_z

        if hidden_z is not None:
            attn_output *= hidden_z

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module, following Llama attention module. This module inherits from `LlamaAttention`
    as the weights of the module stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal with padding tokens
    in case the input contains any of them. Additionally, for sliding window attention, we apply SWA only to the bottom
    config.max_window_layers layers.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal('2.1.0')

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        head_z=None,
        head_layer_z=None,
        hidden_z=None,
        qk_head_dim_z=None,
        vo_head_dim_z=None,
        kv_head_z=None,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        if qk_head_dim_z is not None:
            query_states = query_states.mul(qk_head_dim_z)
            query_states = query_states.to(key_states.dtype)
        if vo_head_dim_z is not None:
            value_states = value_states.mul(vo_head_dim_z)
            value_states = value_states.to(key_states.dtype)
        
        if kv_head_z is not None: # only work for MHA model
            if head_z is not None:
                kv_head_z = kv_head_z * head_z.reshape(kv_head_z.shape)
            '''
            # 0 ~ 1
            ratio = (kv_head_z.shape[-1] - kv_head_z.sum(-1)) / (kv_head_z.shape[-1] - 1)
            ratio = ratio.clamp(0, 1)
            ratio = ratio.unsqueeze(-1).expand_as(kv_head_z)
            # 1 ~ 0
            ratio = 1 - ratio
            kv_head_z_weights = torch.repeat_interleave(ratio.reshape(-1), self.head_dim)
            '''
            kv_head_z_weights = torch.repeat_interleave(kv_head_z.reshape(-1), self.head_dim)

            selected_key_states = self.kv_layer[0](hidden_states)
            selected_key_states = selected_key_states.reshape(bsz, q_len, kv_head_z.shape[0], -1)
            selected_key_states = torch.repeat_interleave(selected_key_states, kv_head_z.shape[-1], 2)
            selected_key_states = selected_key_states.reshape(bsz, q_len, -1)
            selected_value_states = self.kv_layer[1](hidden_states)
            selected_value_states = selected_value_states.reshape(bsz, q_len, kv_head_z.shape[0], -1)
            selected_value_states = torch.repeat_interleave(selected_value_states, kv_head_z.shape[-1], 2)
            selected_value_states = selected_value_states.reshape(bsz, q_len, -1)

            key_states = key_states * kv_head_z_weights + selected_key_states * (1 - kv_head_z_weights)
            value_states = value_states * kv_head_z_weights + selected_value_states * (1 - kv_head_z_weights)

            key_states = key_states.to(query_states.dtype)
            value_states = value_states.to(query_states.dtype)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # if (
        #     self.config.use_sliding_window
        #     and getattr(self.config, "sliding_window", None) is not None
        #     and self.layer_idx >= self.config.max_window_layers
        # ):
        #     sliding_window = self.config.sliding_window
        # else:
        #     sliding_window = None

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=None,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )
        # attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim).contiguous()
        attn_output = attn_output.transpose(1, 2).contiguous()
        if head_z is not None:
            attn_output *= head_z.unsqueeze(-1)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if head_layer_z is not None:
            attn_output *=  head_layer_z

        if hidden_z is not None:
            attn_output *= hidden_z

        attn_weights = None
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        if config._attn_implementation == 'flash_attention_2':
            self.self_attn = LlamaFlashAttention2(config=config, layer_idx=layer_idx)
            logger.warning_once(
                    "using flash attention 2 backend..."
                )
        else:
            self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
            logger.warning_once(
                    "using eager attention backend..."
                )

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def prune_params(self, zs_layer):
        self.self_attn.prune_params(zs_layer)
        self.mlp.prune_params(zs_layer)

        if self.self_attn.q_proj is None:
            self.input_layernorm = None
        if self.mlp.gate_proj is None:
            self.post_attention_layernorm = None
        
        if "hidden_z" in zs_layer:
            hidden_z = zs_layer["hidden_z"]
            if self.input_layernorm is not None:
                self.input_layernorm.prune_params(hidden_z)
            if self.post_attention_layernorm is not None:
                self.post_attention_layernorm.prune_params(hidden_z)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            intermediate_z=None,
            mlp_z=None,
            head_z=None,
            head_layer_z=None,
            hidden_z=None,
            qk_head_dim_z=None,
            vo_head_dim_z=None,
            kv_head_z=None
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states
        present_key_value = past_key_value
        self_attn_weights = None
        if self.input_layernorm is not None:
            hidden_states = self.input_layernorm(hidden_states, hidden_z=hidden_z)
            # Self Attention
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                head_z=head_z,
                head_layer_z=head_layer_z,
                hidden_z=hidden_z,
                qk_head_dim_z=qk_head_dim_z,
                vo_head_dim_z=vo_head_dim_z,
                kv_head_z=kv_head_z,
            )
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        if self.post_attention_layernorm is not None:
            hidden_states = self.post_attention_layernorm(hidden_states, hidden_z=hidden_z)
            hidden_states = self.mlp(
                hidden_states, 
                intermediate_z=intermediate_z, 
                mlp_z=mlp_z, 
                hidden_z=hidden_z, 
                head_layer_z=head_layer_z
            )
            hidden_states = residual + hidden_states

        residual = hidden_states

        outputs = (residual,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # if is_flash_attn_2_available():
        #     self.config._attn_implementation = 'flash_attention_2'
        # else:
        #     self.config._attn_implementation = 'eager'
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.l0_module = None
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def init_l0_module(self, l0_config):# omegaconf
        self.l0_module = L0Module(l0_config, device=self.device)
        # layer, heads, len_group
        if 'kv_head' in self.l0_module.masks:
            kv_head_z = self.l0_module.masks['kv_head'].sample_z()
            for idx, layer in enumerate(self.layers):
                layer.self_attn.kv_layer = nn.ModuleList()
                head_dim = self.config.hidden_size // self.config.num_attention_heads
                target_kv_dim = head_dim * kv_head_z.shape[1]
                layer.self_attn.kv_layer.append(nn.Linear(self.config.hidden_size, target_kv_dim, bias=self.config.attention_bias))
                layer.self_attn.kv_layer.append(nn.Linear(self.config.hidden_size, target_kv_dim, bias=self.config.attention_bias))
                group_size = int(self.config.num_attention_heads / self.l0_module.target_model_info.num_key_value_heads)
                layer.self_attn.kv_layer[0].weight.data = layer.self_attn.convert_mha_to_gqa(
                    layer.self_attn.k_proj.weight.data, self.config.num_attention_heads, group_size
                    )
                layer.self_attn.kv_layer[1].weight.data = layer.self_attn.convert_mha_to_gqa(
                    layer.self_attn.v_proj.weight.data, self.config.num_attention_heads, group_size
                    )
    def prune_params(self, zs=None):
        if zs is None:
            self.l0_module.eval()
            zs = self.l0_module(calculate_lagrangian=False)
        if "hidden_z" in zs:
            hidden_z = zs["hidden_z"]
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            self.norm.prune_params(hidden_z=hidden_z)
            self.embed_tokens.weight.data = self.embed_tokens.weight.data.mul(hidden_z)
            self.embed_tokens.weight = nn.Parameter(
                self.embed_tokens.weight.index_select(1, remaining_index).clone()
            )
            self.embed_tokens.embedding_dim = len(remaining_index)
        for i, layer in enumerate(self.layers):
            zs_layer = self.get_zs_layer(zs, i)
            layer.prune_params(zs_layer)
        return zs

    def get_zs_layer(self, zs, layer_idx):
        zs_layer = {}
        if zs is not None:
            for key in zs:
                if key == "hidden_z": zs_layer["hidden_z"] = zs["hidden_z"]
                else: zs_layer[key] = zs[key][layer_idx] 
        return zs_layer


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        pruned_steps : int = 0,
        **zs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        use_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache) and not self.training:
            use_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.l0_module is not None:
            assert zs == {}, "zs should be empty when using L0Module"
            zs = self.l0_module(calculate_lagrangian=False, pruned_steps=pruned_steps)
        # emb mask
        if "hidden_z" in zs:
            inputs_embeds = inputs_embeds.mul(zs['hidden_z'])
        
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds
    
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for idx, decoder_layer in enumerate(self.layers):
            zs_layer = self.get_zs_layer(zs, idx)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    zs_layer.get('intermediate_z'),
                    zs_layer.get('mlp_z'),
                    zs_layer.get('head_z'),
                    zs_layer.get('head_layer_z'),
                    zs_layer.get('hidden_z'),
                    zs_layer.get('qk_head_dim_z'),
                    zs_layer.get('vo_head_dim_z'),
                    zs_layer.get('kv_head_z')
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **zs_layer,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states, hidden_z=zs.get("hidden_z", None))

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        ), zs

    # Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


_init_weights = True
@contextmanager
def no_init_weights(_enable=True):
    global _init_weights
    old_init_weights = _init_weights
    if _enable:
        _init_weights = False
    try:
        yield
    finally:
        _init_weights = old_init_weights

class DeepSeekForL0Prune(LlamaPreTrainedModel):
    # _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def init_l0_module(self, l0_config):
        self.model.init_l0_module(l0_config)
    def prune_params(self, zs=None):
        zs = self.model.prune_params(zs=zs)
        if "hidden_z" in zs:
            hidden_z = zs["hidden_z"]
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            self.lm_head.weight.data = self.lm_head.weight.data.mul(hidden_z) 
            dtype = self.lm_head.weight.data.dtype
            self.lm_head = prune_linear_layer(self.lm_head, remaining_index, dim=1)
            self.lm_head = self.lm_head.to(dtype)
        # update config
        self.config.hidden_size = self.model.l0_module.target_model_info.hidden_size
        self.config.num_hidden_layers = self.model.l0_module.target_model_info.num_layers
        self.config.num_attention_heads = self.model.l0_module.target_model_info.num_attention_heads
        self.config.num_key_value_heads = self.model.l0_module.target_model_info.num_key_value_heads
        self.config.intermediate_size = self.model.l0_module.target_model_info.intermediate_size
        self.model.l0_module = None

    def convert_format(self):
        if self.model.l0_module is not None:
            self.prune_params()
        
        layers = nn.ModuleList()
        import copy
        for layer in self.model.layers:
            if layer.input_layernorm is None:
                continue
            layers.append(copy.deepcopy(layer))
        self.model.layers = layers
        for idx, layer in enumerate(self.model.layers):
            layer.self_attn.layer_idx = idx
        
    def save_config(self, path):
        self.config.architectures[0] = 'LlamaForCausalLM'
        self.config.save_pretrained(path)

    def load_l0_parameters(self, path):
        # l0_config
        l0_config_path = os.path.join(path, 'l0_config.yaml')
        if not os.path.exists(l0_config_path):
            print0('There is no l0 config file')
            return
        l0_cfg = om.load(l0_config_path)
        self.init_l0_module(l0_config=l0_cfg.model)
        # bins 
        l0_state_dict = OrderedDict()
        param_file_format = 'bin'
        param_files = glob(path+'/*model*.bin')
        if not param_files:
            param_files = glob(path+'/*model*.safetensors')
            param_file_format = 'safetensors'
            # safe tensors
        for param_file in param_files:
            if param_file_format == 'bin':
                params = torch.load(param_file)
            else:
                params = load_file(param_file)
            for param in tqdm(params):
                if 'l0_module' in param:
                    new_param = param.replace('model.l0_module.', '')
                    l0_state_dict.update({new_param: params.get(param)})
                if 'kv_layer' in param:
                    self.load_state_dict(params, strict=False)
        # print0(l0_state_dict)
        self.model.l0_module.load_state_dict(l0_state_dict)
        print0('L0 parameters loaded.')
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=False,
                proxies=None,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder="",
                _from_auto=False,
                _from_pipeline=None,
                **kwargs,
            )
        else:
            model_kwargs = kwargs
        
        # if hasattr(config, "quantization_config") and config.quantization_config['load_in_4bit']:
        #     try:
        #         from .quantizer import init_model_weight_int4
        #         from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map
        #         from accelerate.utils import CustomDtype
        #         from accelerate.utils import get_balanced_memory
        #     except ImportError:
        #         raise ImportError(f"Needs import model weight init func to run quantize.") 
        #     # Instantiate model.
        #     init_contexts = [no_init_weights(_enable=True)]
        #     init_contexts.append(init_empty_weights())
        #     with ContextManagers(init_contexts):
        #         model = cls(config)
            
        #     model_file = os.path.join(pretrained_model_name_or_path, 'pytorch_model.bin')
        #     state_dict = torch.load(model_file, map_location="cpu") 
        #     model.is_quantized = True
            
        #     device_map = kwargs.pop("device_map", None)
        #     torch_dtype = kwargs.pop("torch_dtype", None)
            
        #     if device_map is not None:
        #         kwargs = {"no_split_module_classes": model._no_split_modules}
        #         target_dtype = CustomDtype.INT4
        #         max_memory = get_balanced_memory(
        #             model,
        #             dtype=target_dtype,
        #             low_zero=(device_map == "balanced_low_0"),
        #             max_memory=None,
        #             **kwargs,
        #         )
        #         kwargs["max_memory"] = max_memory
        #         device_map = infer_auto_device_map(model, dtype=target_dtype, **kwargs)
                
        #     model = init_model_weight_int4(config, model, state_dict)
            
        #     # Set model in evaluation mode to deactivate DropOut modules by default
        #     model.eval()
        #     # If it is a model with generation capabilities, attempt to load the generation config
        #     if model.can_generate():
        #         try:
        #             model.generation_config = GenerationConfig.from_pretrained(
        #                 pretrained_model_name_or_path,
        #                 cache_dir=cache_dir,
        #                 force_download=force_download,
        #                 resume_download=False,
        #                 proxies=None,
        #                 local_files_only=local_files_only,
        #                 token=token,
        #                 revision=revision,
        #                 subfolder="",
        #                 _from_auto=False,
        #                 _from_pipeline=None,
        #                 **kwargs,
        #             )
        #         except (OSError, TypeError):
        #             logger.info(
        #                 "Generation config file not found, using a generation config created from the model config."
        #             )
        #             pass
            
        #     if device_map is not None:
        #         dispatch_model(model, device_map=device_map)
            
        #     return model
        return super(DeepSeekForL0Prune, cls).from_pretrained(pretrained_model_name_or_path, *model_args, device_map=None,
                config=config, cache_dir=cache_dir, ignore_mismatched_sizes=ignore_mismatched_sizes, 
                force_download=force_download, local_files_only=local_files_only, token=token, revision=revision, 
                use_safetensors=use_safetensors, **kwargs)   

    def forward_for_train(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            pruned_steps: int = 0,
            **zs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs, zs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            pruned_steps=pruned_steps,
            **zs
        )

        hidden_states = outputs.last_hidden_state

        logits = self.lm_head(hidden_states)
        lag_loss = None
        if self.model.l0_module is not None:
            lag_loss, _ = self.model.l0_module(calculate_lagrangian=True, pruned_steps=pruned_steps)
        lm_loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            softmax_normalizer = shift_logits.max(-1).values ** 2
            z_loss = self.config.z_loss_weight * softmax_normalizer.mean()
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            lm_loss = loss_fct(shift_logits, shift_labels) + z_loss
        loss = 0
        if lag_loss is not None:
            loss += lag_loss
        if lm_loss is not None:
            loss += lm_loss
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), zs, lm_loss, lag_loss
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        pruned_steps: int = 0,
        **zs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = True

        outputs, zs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            pruned_steps=pruned_steps,
            **zs
        )

        hidden_states = outputs.last_hidden_state

        logits = self.lm_head(hidden_states)
        lm_loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            softmax_normalizer = shift_logits.max(-1).values ** 2
            z_loss = self.config.z_loss_weight * softmax_normalizer.mean()
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            lm_loss = loss_fct(shift_logits, shift_labels) + z_loss
        loss = lm_loss
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs