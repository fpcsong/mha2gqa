import math
import sys
import os
import torch
from torch import nn
import random
from torch.autograd import Variable
import deepspeed
from deepspeed.runtime.zero import GatheredParameters
import torch.nn.functional as F

from argparse import Namespace as NS
from typing import Any, List


def print0(*message):
    """If distributed is initialized, print only on rank 0."""
    if int(os.environ.get('LOCAL_RANK', -1)) in [-1, 0]:
            print(*message, flush=True)

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6

def lp_losses(curr_sparsity, curr_target_sparsity, target_sparsity, reduction='mean'):
    
    base_loss = F.l1_loss(curr_sparsity, curr_target_sparsity, reduction='none')
    if reduction == 'mean':
        l1_loss = base_loss.mean()
        l2_loss = torch.pow(base_loss + epsilon, 2).mean()
        l05_loss = torch.pow(base_loss + epsilon, 0.5).mean()
    else:
        l1_loss = base_loss.sum()
        l2_loss = torch.pow(base_loss + epsilon, 2).sum()
        l05_loss = torch.pow(base_loss + epsilon, 0.5).sum()
    # return l1_loss + l2_loss + l05_loss
    loss = l1_loss + l2_loss
    # scale = abs(target_sparsity - curr_sparsity) / (target_sparsity - curr_target_sparsity + 1e-2)
    # print0(scale)
    # loss *= scale.mean()
    return loss

class STEhradtanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return F.hardtanh(inp, min_val=0., max_val=1.)
    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input.clamp_(-1, 1)
        # del the following two lines to enable ste tanh
        # grad_input[inp < 0] = 0
        # grad_input[inp > 1] = 0
        return grad_input
custom_hardtanh = STEhradtanh.apply

class Mask(nn.Module):
    def __init__(self, 
                 name: str,
                 mask_shape: List, 
                 num_params_per_mask: int, 
                 mask_output_shape: List, 
                 target_sparsity: float,
                 target_mask_size: int,
                 device: str,
                 eval_target_model: bool=True,
                 alpha: float=1e-4) -> None:
        super().__init__()
        self.name = name
        self.num_params_per_mask = num_params_per_mask
        self.mask_output_shape = mask_output_shape
        self.target_sparsity=target_sparsity

        self.droprate_init = 0.05
        self.std = 1.
        self.max_std = 1.
        self.min_std = 0.1
        self.temperature = 1./3.
        # self.max_temperature = 1./3.
        # self.min_temperature = 1./5.
        self.device = device
        self.alpha = alpha
        self.max_noise = 0.5
        self.noise = 0.1
        self.offset_gate = 0.995
        self.magic_number = math.log(self.offset_gate) - math.log(1-self.offset_gate)

        self.z_loga = self.initialize_mask(mask_shape) 
        self.mask_size = self.z_loga.shape[-1] # the full size of each unit
        self.target_mask_size = target_mask_size
        self.eval_target_model = eval_target_model
        
    def param_init_fn(self, module):
        """ Initialize the parameters for masking variables. """
        # mean = math.log(1 - self.droprate_init) - math.log(self.droprate_init)
        mean = 6 * self.temperature * self.alpha
        if isinstance(module, nn.Parameter):
            module.data.normal_(mean, self.alpha/10)
        else:
            for tensor in module.parameters():
                tensor.data.normal_(mean, self.alpha/10)
        
    def initialize_mask(self, mask_shape: List):
        """ Initialize the parameters for masking variables. """
        z_loga = nn.Parameter(torch.ones(*mask_shape, device=self.device), requires_grad=True)
        self.param_init_fn(z_loga)
        z_loga = z_loga.to(torch.float32)
        return z_loga
    # '''
    def cdf_qz(self, z_loga = None):
        if z_loga is None:
            z_loga = self.z_loga.to(torch.float32)
        fake_inputs = torch.ones_like(z_loga.data).to(z_loga.device) /self.alpha
        y = torch.sigmoid(
            (-self.magic_number * self.temperature - z_loga * fake_inputs / self.temperature)
            ).clamp(epsilon, 1-epsilon)
        return y

    def calculate_expected_score_sparsity(self):
        # expected_num_nonzeros
        score = 1 - self.cdf_qz()
        sparsity = 1 - score.sum(-1) / self.mask_size
        sparsity = sparsity.to(torch.float32)
        score = score.to(torch.float32)
        return score, sparsity

    def inv_cdf_qz(self, z_loga=None):
        if z_loga is None:
            z_loga = self.z_loga.to(torch.float32)
        fake_inputs = torch.ones_like(z_loga.data).to(z_loga.device) /self.alpha
        y = torch.sigmoid(
            (self.magic_number * self.temperature + z_loga * fake_inputs / self.temperature)
        ).clamp(epsilon, 1-epsilon) * (z_loga > 0).long()
        return y

    def calculate_inv_expected_score_sparsity(self):
        # expected_num_zeros
        score = 1 - self.inv_cdf_qz()# num ones
        inv_sparsity = 1 - score.sum(-1) / self.mask_size
        inv_sparsity = inv_sparsity.to(torch.float32)
        score = score.to(torch.float32)
        return score, inv_sparsity
    
    def get_eps(self, size: List):

        eps = torch.FloatTensor(size).uniform_(0.5 - self.noise, 0.5 + self.noise)
        return eps
    
    def sample_z(self, keepdim=False, nonoise=False):
        
        fake_inputs = torch.ones_like(self.z_loga.data).to(self.z_loga.device) /self.alpha

        # eps = torch.normal(0, self.std, self.z_loga.shape).to(self.z_loga.device)
        # y = torch.sigmoid(
        #     (eps + self.z_loga * fake_inputs) / self.temperature
        #     )

        eps = self.get_eps(torch.FloatTensor(*self.z_loga.shape)).to(self.z_loga.device)
        if nonoise:
            eps = 0.5 * torch.ones_like(eps)
        y = torch.sigmoid(
            # 噪声独立，不受温度影响
                torch.log(eps) - torch.log(1 - eps) + self.z_loga * fake_inputs / self.temperature
            )
        # (1 / sigmoid(6) - 1) * 2 + 1, because lower_bound in constrain sparsity is -6
        y = (y - 0.5) * 1.005 + 0.5
        # add STE
        z = custom_hardtanh(y)
        if not keepdim:
            z = z.reshape(*self.mask_output_shape)
        z = z.to(torch.float32)
        return z
    def detsoft_z(self):
        fake_inputs = torch.ones_like(self.z_loga.data).to(self.z_loga.device) /self.alpha
        y = torch.sigmoid(
            (self.z_loga * fake_inputs) / self.temperature
            )
        y = (y - 0.5) * (1 + 2 * epsilon) + 0.5
        z = F.hardtanh(y, min_val=0., max_val=1.).reshape(*self.mask_output_shape)
        return z
    def _deterministic_z(self, z_loga):
        # Following https://github.com/asappresearch/flop/blob/e80e47155de83abbe7d90190e00d30bfb85c18d5/flop/hardconcrete.py#L8 line 103
        if self.target_mask_size is None or not self.eval_target_model:
            expected_score = 1 - self.cdf_qz(z_loga)
            expected_num_nonzeros = expected_score.sum()
            expected_num_zeros = z_loga.nelement() - expected_num_nonzeros.item()
        else:
            expected_num_zeros = self.mask_size - self.target_mask_size 
        try:
            num_zeros = round(expected_num_zeros)
        except:
            print0("num of zeros is nan....")
            sys.exit()
        fake_inputs = torch.ones_like(z_loga.data).to(z_loga.device) /self.alpha
        soft_mask = torch.sigmoid(z_loga * fake_inputs / self.temperature)
        if num_zeros > 0:
            if soft_mask.ndim == 0:
                soft_mask = torch.tensor(0).to(self.z_loga.device)
            else:
                _, indices = torch.topk(soft_mask, k=num_zeros, largest=False)
                soft_mask[indices] = 0.
                # soft_mask = (soft_mask + (soft_mask > 0).long()).clamp(0, 1)
        # print0(soft_mask.nelement(), num_zeros, soft_mask.sum() / (soft_mask > 0).long().sum())
        soft_mask = soft_mask.to(torch.float32)
        return soft_mask
    
    def deterministic_z(self):
        if self.z_loga.ndim == 1:
            z = self._deterministic_z(self.z_loga).reshape(*self.mask_output_shape)
        else:
            z_loga = self.z_loga.reshape(-1, self.z_loga.shape[-1])
            z = []
            for i in range(z_loga.shape[0]):
                z_ = self._deterministic_z(z_loga[i])
                z.append(z_)
            z = torch.stack(z).reshape(*self.mask_output_shape)
        return z
    
    def forward(self):
        print0(self.training)
        func = self.sample_z if self.training else self.deterministic_z
        z = func(self.z_loga).reshape(self.mask_output_shape)
        return z

    def constrain_parameters(self):
        self.z_loga.data.clamp_(min=-8 * self.alpha, max= 8 * self.alpha)
    
    def constrain_sparsity(self, pruned_steps, warmup_steps, max_prune_steps, curr_sparsity=None):
        # ATTENTION! warmup ratio may > 1!
        _min = -10 * self.alpha * self.temperature
        _max = 10 * self.alpha * self.temperature
        warmup_ratio = min(1, pruned_steps/warmup_steps)
        curr_target_sparsity = self.target_sparsity * warmup_ratio
        scale = math.exp(math.sqrt(pruned_steps/max_prune_steps) * 1)
        base_ratio = 0.999 if pruned_steps < max_prune_steps else 0.8
        if curr_sparsity is not None:
            k = int(self.mask_size * curr_target_sparsity + 1)
            lk = int(min(k, self.mask_size - self.target_mask_size))
            if curr_target_sparsity > curr_sparsity.mean().item():
                new_rescale_ratio = 1 - (curr_target_sparsity - curr_sparsity.mean().item()) * scale
                rescale_ratio = max(0.6, min(base_ratio, new_rescale_ratio))
                with torch.no_grad():
                    if lk >= 1:
                        # sort is faster than kthvalue
                        lvalues = self.z_loga.data.sort(-1)[0][..., lk-1].unsqueeze(-1)
                        lvalues =lvalues.expand_as(self.z_loga.data).to(self.z_loga.dtype)
                        updated_zloga = (_min + (self.z_loga.data - _min) * rescale_ratio).clone()
                        updated_zloga.clamp_(min=_min)
                        lmask = (self.z_loga.data <= lvalues).long().float() * (lvalues > _min).long().float()
                        self.z_loga.data = lmask * updated_zloga + (1-lmask) * self.z_loga.data
                # print0('\nlk ratio ', self.name, lk, rescale_ratio)
            new_rescale_ratio = 1 + (curr_target_sparsity - curr_sparsity.mean().item()) * scale
            rescale_ratio = min(base_ratio, new_rescale_ratio)
            rk = min(self.mask_size, lk+1)
            # print0('\nrk ratio : ',rk, rescale_ratio)
            # print0('\ntgt cur', curr_target_sparsity, curr_sparsity.mean().item())
            with torch.no_grad():
                if rk < self.mask_size:
                    rvalues = self.z_loga.data.sort(-1)[0][..., rk-1].unsqueeze(-1)
                    rvalues = rvalues.expand_as(self.z_loga.data).to(self.z_loga.dtype)
                    updated_zloga = (_max + (self.z_loga.data - _max) * rescale_ratio).clone()
                    updated_zloga.clamp_(max=_max)
                    rmask = (self.z_loga.data >= rvalues).long().float() * (rvalues < _max).long().float()
                    self.z_loga.data = rmask * updated_zloga + (1-rmask) * self.z_loga.data

def calculate_params(vocab_size, d_model, n_layers, n_heads, n_kv_heads, intermediate_size):
    # Embedding layer
    embedding_params = vocab_size * d_model * 2

    # Q (query) params
    q_proj_params = d_model * d_model

    # K (key) params
    k_proj_params = d_model * d_model / n_heads * n_kv_heads

    # V (value) params
    v_proj_params = d_model * d_model / n_heads * n_kv_heads

    # Attention output projection
    attn_output_params = d_model * d_model

    # Attention layer params
    attn_layer_params = q_proj_params + k_proj_params + v_proj_params + attn_output_params

    # Feedforward parameters
    ffw_up_params = d_model * intermediate_size * 2  # up gate
    ffw_down_params = intermediate_size * d_model  # down
    ffw_layer_params = ffw_up_params + ffw_down_params

    # Layer normalization params (2 per transformer layer)
    layer_norm_params = 2 * (2 * d_model)

    # Total parameters for one layer
    per_layer_params = attn_layer_params + ffw_layer_params + layer_norm_params

    # Total transformer block parameters
    total_transformer_params = per_layer_params * n_layers

    # Total model parameters
    total_params = embedding_params + total_transformer_params

    return total_params

class L0Module(nn.Module):
    def __init__(self, cfg, device):
        super(L0Module, self).__init__()

        # base and target model info
        n_matrix_mlp = 2 if "pythia" in cfg.name else 3
        self.base_model_info = self.set_model_info(cfg, n_matrix_mlp=n_matrix_mlp) 
        l0_module_cfg = cfg.l0_module
        self.target_model_info = None
        target_model_cfg = getattr(l0_module_cfg, "target_model", None)
        if target_model_cfg is not None:
            self.target_model_info = self.set_model_info(target_model_cfg, n_matrix_mlp=n_matrix_mlp)
        
        # l0 config
        self.pruning_modules = l0_module_cfg.pruning_modules        
        self.start_sparsity = l0_module_cfg.start_sparsity 
        self.lagrangian_warmup_steps = l0_module_cfg.lagrangian_warmup_steps
        self.max_prune_steps = l0_module_cfg.max_prune_steps
        # assert self.lagrangian_warmup_steps * 2 <= self.max_prune_steps
        self.sep_warmup_steps = {}
        for pruning_module in self.pruning_modules:
            self.sep_warmup_steps[pruning_module] = self.lagrangian_warmup_steps


        self.sft_steps = l0_module_cfg.sft_steps
        self.device = device
        self.eval_target_model = l0_module_cfg.get("eval_target_model", True)
        self.alpha = l0_module_cfg.get('alpha', 1e-4)
        self.sparsity_pool = l0_module_cfg.get('sparsity_pool', 0.1)
        
        # l0 params
        self.lambdas = {}
        self.lambdas["lambda_1"] = torch.nn.Parameter(torch.tensor(0.0, device=device))
        self.lambdas["lambda_2"] = torch.nn.Parameter(torch.tensor(0.0, device=device))
        self.masks = {}
        self.pre_sparsity = {}
        self.last_prune_step = -1
        self.aux_loss_weight = 0.01
        for pruning_module in self.pruning_modules:
            self.initialize_one_module(pruning_module)
            self.pre_sparsity[pruning_module] = None
        self.masks = torch.nn.ModuleDict(self.masks)
        self.lambdas = torch.nn.ParameterDict(self.lambdas)
        
        # config after initialization
        self.prunable_model_size = self.calculate_prunable_model_size(self.base_model_info)
        if target_model_cfg is not None:
            self.prunable_target_model_size = self.calculate_prunable_model_size(self.target_model_info)
            self.target_sparsity = 1 - self.prunable_target_model_size / self.prunable_model_size
        else:
            self.target_sparsity = l0_module_cfg.target_sparsity

        print0("********** Initializing L0 Module **********") 
        for pruning_module in self.pruning_modules:
            print0(f"***** {pruning_module} *****")
            print0(f"z.shape", self.masks[pruning_module].z_loga.shape)
            print0(f"size", self.masks[pruning_module].mask_size)
        print0(f"prunable model size: {self.prunable_model_size/1e9}B")
        if target_model_cfg is not None:
            print0(f"prunable_target_model_size: {self.prunable_target_model_size/1e9}B")
        print0(f"max_prune_steps: {self.max_prune_steps}")
        print0(f"lagrangian_warmup_steps: {self.lagrangian_warmup_steps}")
        
    
    def set_model_info(self, cfg, n_matrix_mlp):
        ns = NS() 
        ns.hidden_size = cfg.d_model
        ns.intermediate_size = cfg.intermediate_size
        ns.num_attention_heads = cfg.n_heads
        ns.num_key_value_heads = cfg.n_kv_heads
        ns.mlp_num_per_layer = 1
        ns.dim_per_head = ns.hidden_size // ns.num_attention_heads 
        ns.num_layers = cfg.n_layers
        ns.vocab_size = cfg.vocab_size

        ns.params_per_head_layer = ns.hidden_size * ns.hidden_size * 2 # vo
        ns.params_per_head_layer += ns.hidden_size * ns.dim_per_head * ns.num_key_value_heads * 2 # qk
        ns.params_per_head =  ns.params_per_head_layer // ns.num_attention_heads
        ns.params_per_mlp_layer = ns.hidden_size * ns.intermediate_size * n_matrix_mlp
        ns.params_per_intermediate_dim = ns.params_per_mlp_layer // ns.intermediate_size

        ns.full_model_size = (ns.params_per_head_layer + ns.params_per_mlp_layer) * ns.num_layers
        return ns
        
    def calculate_prunable_model_size(self, ns: NS):
        return calculate_params(
            ns.vocab_size, 
            ns.hidden_size, 
            ns.num_layers, 
            ns.num_attention_heads, 
            ns.num_key_value_heads, 
            ns.intermediate_size
            )

    def initialize_one_module(self, module_name: str):
        func_name = f"initialize_{module_name}"
        try:
            method = getattr(self, func_name)
        except AttributeError:
            raise NotImplementedError("Instance `{}` does not implement `{}`".format(self, func_name))
        method()
            

    def initialize_kv_head(self):
        # for MHA->GQA
        num_key_value_heads = self.target_model_info.num_key_value_heads
        # target model MUST be a GQA-based model
        len_group = int(self.base_model_info.num_attention_heads / num_key_value_heads)
        assert self.target_model_info.num_attention_heads % num_key_value_heads == 0
        assert self.base_model_info.num_attention_heads % num_key_value_heads == 0
        # only for MHA model
        assert self.base_model_info.num_attention_heads == self.base_model_info.num_key_value_heads

        mask_shape = [self.base_model_info.num_layers * num_key_value_heads * len_group]

        num_params_per_mask = self.base_model_info.params_per_head
        mask_output_shape = [self.base_model_info.num_layers, num_key_value_heads, len_group]
        
        target_head_sparsity = None; pd = {} ; target_mask_size=None; 
        if self.target_model_info is not None:
            # target_head_sparsity = 1 - self.target_model_info.num_key_value_heads / self.base_model_info.num_key_value_heads
            # target_mask_size = 1 # each group only keep one kv head
            target_head_sparsity = 1
            target_mask_size = 0
            pd = {"lambda_1_kv_head": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_kv_head": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        head_mask = Mask(name="kv_head",
                         mask_shape=mask_shape,
                         num_params_per_mask=num_params_per_mask,
                         mask_output_shape=mask_output_shape,
                         target_sparsity=target_head_sparsity,
                         target_mask_size=target_mask_size,
                         device=self.device,
                         eval_target_model=self.eval_target_model,
                         alpha=self.alpha)
        # print0(head_mask.target_sparsity)
        self.masks["kv_head"] = head_mask

    def initialize_qk_head_dim(self): # only campatible when target model info is available
        mask_shape = [self.base_model_info.num_layers, self.base_model_info.num_attention_heads, self.base_model_info.dim_per_head]
        num_params_per_mask = 2 * self.base_model_info.hidden_size
        mask_output_shape = [self.base_model_info.num_layers, self.base_model_info.hidden_size] 
        
        target_qk_head_dim_sparsity = None; pd = {} 
        if self.target_model_info is not None:
            # target_qk_head_dim_sparsity = 1 - self.target_model_info.hidden_size / self.base_model_info.hidden_size
            assert self.target_model_info.hidden_size % self.target_model_info.num_attention_heads == 0
            target_mask_size = self.target_model_info.hidden_size / self.target_model_info.num_attention_heads
            mask_size = self.base_model_info.dim_per_head
            target_qk_head_dim_sparsity = 1 - (target_mask_size / mask_size)
            pd = {"lambda_1_qk_head_dim": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_qk_head_dim": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        qk_head_dim = Mask(name="qk_head_dim",
                         mask_shape=mask_shape,
                         num_params_per_mask=num_params_per_mask,
                         mask_output_shape=mask_output_shape,
                         target_sparsity=target_qk_head_dim_sparsity,
                         target_mask_size=target_mask_size,
                         device=self.device,
                         alpha=self.alpha)
        self.masks["qk_head_dim"] = qk_head_dim

    def initialize_vo_head_dim(self): # only campatible when target model info is available
        mask_shape = [self.base_model_info.num_layers, self.base_model_info.num_attention_heads, self.base_model_info.dim_per_head]
        num_params_per_mask = 2 * self.base_model_info.hidden_size
        mask_output_shape = [self.base_model_info.num_layers, self.base_model_info.hidden_size] 
        
        target_vo_head_dim_sparsity = None; pd = {} 
        if self.target_model_info is not None:
            assert self.target_model_info.hidden_size % self.target_model_info.num_attention_heads == 0
            target_mask_size = self.target_model_info.hidden_size / self.target_model_info.num_attention_heads
            mask_size = self.base_model_info.dim_per_head
            target_vo_head_dim_sparsity = 1 - (target_mask_size / mask_size)
            pd = {"lambda_1_vo_head_dim": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_vo_head_dim": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        vo_head_dim = Mask(name="vo_head_dim",
                         mask_shape=mask_shape,
                         num_params_per_mask=num_params_per_mask,
                         mask_output_shape=mask_output_shape,
                         target_sparsity=target_vo_head_dim_sparsity,
                         target_mask_size=target_mask_size,
                         device=self.device,
                         alpha=self.alpha)
        self.masks["vo_head_dim"] = vo_head_dim
        
    
    def constrain_parameters(self):
        if 'head_layer' in self.masks:
            self.masks['head_layer'].z_loga.data[:2] = 6 * self.alpha
            self.masks['head_layer'].z_loga.data[-2:] = 6 * self.alpha

        for key in self.masks:
            self.masks[key].constrain_parameters()

    def constrain_sparsity(self, pruned_steps):
        for key in self.masks:
            self.masks[key].constrain_sparsity(pruned_steps, self.sep_warmup_steps[key], self.max_prune_steps, self.pre_sparsity[key])


    def decay_noise(self, pruned_steps):
        ratio = 1 - min(pruned_steps / self.max_prune_steps, 1)
        for key in self.masks:
            self.masks[key].noise = self.masks[key].max_noise * ratio

    def calculate_expected_score_sparsity(self):
        expected_scores = {}
        expected_sparsitys = {}
        for key in self.masks:
            score, sparsity = self.masks[key].calculate_expected_score_sparsity()
            expected_scores[key] = score
            expected_sparsitys[key] = sparsity
        return expected_scores, expected_sparsitys
    def calculate_inv_expected_score_sparsity(self):
        expected_scores = {}
        expected_sparsitys = {}
        for key in self.masks:
            score, sparsity = self.masks[key].calculate_inv_expected_score_sparsity()
            expected_scores[key] = score
            expected_sparsitys[key] = sparsity
        return expected_scores, expected_sparsitys
    def transform_scores_for_head(self, expected_scores: dict):
        head_score = expected_scores["head"] # 12 * 12

        head_layer_score = None
        if "head_layer" in expected_scores:
            head_layer_score = expected_scores["head_layer"]
        if head_layer_score is not None:
            head_layer_score = head_layer_score.view(-1, 1)
       
        return head_layer_score, head_score

    def transform_scores_for_mlp(self, expected_scores: dict):
        mlp_score = None
        if "mlp" in expected_scores:
            mlp_score = expected_scores["mlp"]
        if mlp_score is not None:
            mlp_score = mlp_score.unsqueeze(-1)
        intermediate_score = expected_scores["intermediate"]
        return mlp_score, intermediate_score

    def get_expected_num_params(self, expected_scores: dict):
        dtype = list(expected_scores.values())[0].dtype
        device = list(expected_scores.values())[0].device
        num_parameters = 0
        head_score = expected_scores.get('head')# 32 * 32
        if head_score is None:
            head_score = torch.ones(self.base_model_info.num_layers, self.base_model_info.num_attention_heads)
            head_score = head_score.to(device).to(dtype)
        head_layer_score = expected_scores.get('head_layer') # 32
        if head_layer_score is None:
            head_layer_score = torch.ones(self.base_model_info.num_layers)
            head_layer_score = head_layer_score.to(device).to(dtype)
        intermediate_score = expected_scores.get('intermediate') # 32 * 11008
        if intermediate_score is None:
            intermediate_score = torch.ones(self.base_model_info.num_layers, self.base_model_info.intermediate_size)
            intermediate_score = intermediate_score.to(device).to(dtype)
        mlp_score = expected_scores.get('mlp') # 32
        if mlp_score is None:
            mlp_score = torch.ones(self.base_model_info.num_layers)
            mlp_score = mlp_score.to(device).to(dtype)
        head_score = head_layer_score.unsqueeze(-1) * head_score
        int_score = mlp_score.unsqueeze(-1) * intermediate_score
        qk_score = None
        if "qk_head_dim" in expected_scores:
            qk_head_dim_score = expected_scores["qk_head_dim"] # num_layer * hidden_size
            vo_head_dim_score = expected_scores["vo_head_dim"] # num_layer * hidden_size
            qk_head_dim_score = qk_head_dim_score.view(qk_head_dim_score.shape[0], -1)
            vo_head_dim_score = vo_head_dim_score.view(vo_head_dim_score.shape[0], -1)
            head_score = torch.repeat_interleave(head_score, self.base_model_info.dim_per_head, dim=1)

            qk_score = head_score * qk_head_dim_score
            vo_score = head_score * vo_head_dim_score
                
        if "hidden" in expected_scores:
            hidden_score = expected_scores["hidden"]
            
            if qk_score is None:
                num_parameters += torch.outer(hidden_score, head_score.reshape(-1)).sum() * self.masks.head.num_params_per_mask / self.base_model_info.hidden_size
                num_parameters += torch.outer(hidden_score, int_score.reshape(-1)).sum() * self.masks.intermediate.num_params_per_mask / self.base_model_info.hidden_size
            else:
                num_parameters += torch.sum(torch.matmul(hidden_score.reshape(1, -1, 1), qk_score.unsqueeze(1))) * 2
                num_parameters += torch.sum(torch.matmul(hidden_score.reshape(1, -1, 1), vo_score.unsqueeze(1))) * 2
                num_parameters += torch.sum(torch.matmul(hidden_score.reshape(1, -1, 1), int_score.unsqueeze(1))) * 3
        else:
            num_parameters += torch.sum(head_score) * self.masks.head.num_params_per_mask
            num_parameters += torch.sum(int_score) * self.masks.intermediate.num_params_per_mask
        return num_parameters
    
    def get_target_sparsity(self, pruned_steps: int, full_sparsity: float, pruning_module: str=None):

        def _sigmoid(x):
            return math.exp(x) / (1+ math.exp(x))

        target_sparsity = full_sparsity
        if pruning_module is None:
            if getattr(self, "lagrangian_warmup_steps", 0) > 0:
                # linear
                coefficient = min(1, pruned_steps / self.lagrangian_warmup_steps)
                # sqrt
                # coefficient = min(1, math.pow(pruned_steps / self.lagrangian_warmup_steps + epsilon, 0.5))
                # sigmoid
                # coefficient = _sigmoid((pruned_steps / self.lagrangian_warmup_steps) * 10 - 5)
                # ^2
                # coefficient = min(1, math.pow(pruned_steps / self.lagrangian_warmup_steps + epsilon, 2))
                target_sparsity = (target_sparsity - self.start_sparsity) * coefficient + self.start_sparsity
        else:
            coefficient = min(1, pruned_steps / self.sep_warmup_steps[pruning_module])
                # sqrt
                # coefficient = min(1, math.pow(pruned_steps / self.lagrangian_warmup_steps + epsilon, 0.5))
                # sigmoid
                # coefficient = _sigmoid((pruned_steps / self.lagrangian_warmup_steps) * 10 - 5)
                # ^2
                # coefficient = min(1, math.pow(pruned_steps / self.lagrangian_warmup_steps + epsilon, 2))
            target_sparsity = (target_sparsity - self.start_sparsity) * coefficient + self.start_sparsity

        return target_sparsity
    def get_relative_target_sparsity(self, pruned_steps, curr_sparsity, pruning_module):
        full_sparsity = self.masks[pruning_module].target_sparsity
        pre_sparsity = self.pre_sparsity[pruning_module]

        tensor_target_sparsity = self.get_target_sparsity(pruned_steps, full_sparsity, pruning_module)
        tensor_target_sparsity = torch.ones_like(curr_sparsity) * tensor_target_sparsity
        return tensor_target_sparsity.to(torch.float32)

        tensor_full_sparsity = torch.ones_like(curr_sparsity) * full_sparsity

        if pre_sparsity is not None:
            min_tensor_target_sparsity = pre_sparsity
        else:
            min_tensor_target_sparsity = torch.zeros_like(curr_sparsity)

        target_sparsity = (curr_sparsity + self.sparsity_pool).clamp(
            min=min_tensor_target_sparsity,
            max=(tensor_target_sparsity + self.sparsity_pool).clamp(max=tensor_full_sparsity),
            )
        return target_sparsity

    def lagrangian_regularization(self, pruned_steps: int=0):
        def _aux_mean_loss(mask, target_sparsity):
            z = mask.sample_z(keepdim=True)
            z_tensor=z.squeeze()

            target_mean = 1-target_sparsity
            real_mean = torch.mean(z_tensor, dim=-1).to(target_sparsity.dtype)
            mean_loss=F.mse_loss(
                torch.exp(real_mean), 
                torch.exp(target_mean), 
                reduction='sum'
            )
            return mean_loss

        def _aux_var_loss(mask, target_sparsity):
            z = mask.sample_z(keepdim=True)
            z_tensor=z.squeeze()

            target_variance=(1-target_sparsity)*target_sparsity
            real_variance = torch.var(z_tensor, dim=-1, correction=0).to(target_sparsity.dtype)
            # 方差尽可能大
            variance_loss=F.mse_loss(
                real_variance,
                target_variance,
                reduction='sum'
            )
            return variance_loss

        def _lag_loss(expected_sparsity: torch.tensor, curr_target_sparsity: torch.tensor, target_sparsity: torch.tensor, reduction='mean'):
            lagrangian_loss = lp_losses(expected_sparsity, curr_target_sparsity, target_sparsity, reduction=reduction)
            return lagrangian_loss
        
        def reduction_map(pruning_module):
            # if pruning_module == 'hidden':
            #     return 'sum'
            return 'mean'
        target_sparsity = self.get_target_sparsity(pruned_steps, self.target_sparsity)
        expected_scores, expected_sparsitys = self.calculate_expected_score_sparsity()
        return_v = {}
        if self.target_model_info is None:
            expected_size = self.get_expected_num_params(expected_scores) #! calculate \bar s
            expected_sparsity = 1 - expected_size / self.prunable_model_size
            lagrangian_loss = _lag_loss(expected_sparsity, expected_sparsity, target_sparsity)
            return_v = {"expected_sparsity": expected_sparsity.item(), "target_sparsity": target_sparsity}
            for key in expected_sparsitys:
                return_v[f"expected_{key}_sparsity"] = expected_sparsitys[key].mean().item()
        else:
            lagrangian_loss = 0
            return_v = {}
            for pruning_module in self.pruning_modules:
                expected_ts = expected_sparsitys[pruning_module]
                ts = self.get_relative_target_sparsity(
                    pruned_steps,
                    expected_ts.detach(),
                    pruning_module
                    )
                self.pre_sparsity[pruning_module] = expected_ts.detach()

                module_lag_loss = _lag_loss(
                    expected_ts, 
                    ts, 
                    self.masks[pruning_module].target_sparsity,
                    reduction=reduction_map(pruning_module))
                lagrangian_loss += module_lag_loss

                # aux_mean_loss = _aux_mean_loss(self.masks[pruning_module], ts)
                # aux_var_loss = _aux_var_loss(self.masks[pruning_module], ts)
                # lagrangian_loss += aux_var_loss * min(1, (pruned_steps / self.sep_warmup_steps[pruning_module]) **4)

        return lagrangian_loss
 
    def forward(self, calculate_lagrangian: bool = False, pruned_steps: int = 1e8):
        self.constrain_parameters()

        if calculate_lagrangian:
            if self.training:
                if pruned_steps != self.last_prune_step:
                    self.last_prune_step = pruned_steps
            lagrangian_loss = self.lagrangian_regularization(pruned_steps)
            if pruned_steps > self.max_prune_steps or pruned_steps < self.sft_steps:
                lagrangian_loss *= 0.
            return lagrangian_loss, {}
        
        zs = {f"{pruning_module}_z": [] for pruning_module in self.pruning_modules}
        
        if self.training and pruned_steps < self.sft_steps:
            return {}
        
        if self.training and pruned_steps < self.max_prune_steps:
            for pruning_module in self.pruning_modules:
                mask = self.masks[pruning_module]
                z = mask.sample_z()
                zs[f"{pruning_module}_z"] = z
            return zs
        self.training = False
        with torch.no_grad():
            for pruning_module in self.pruning_modules:
                mask = self.masks[pruning_module]
                z = mask.deterministic_z()
                zs[f"{pruning_module}_z"] = z
        return zs

def test_l0_module():
    from omegaconf import OmegaConf as om 
    cfg = om.load("config/deepseek-500m-20h.yaml")
    cfg.model.l0_module.pruning_modules = ["layer", "head", "intermediate", "hidden"]
    l0_module = L0Module(cfg.model, "cpu")
    l0_module.lagrangian_warmup_steps=10

    print(l0_module.masks['head'].sample_z())
    print(l0_module.masks['head'].calculate_expected_score_sparsity()[1])
    exit(0)
    
    # test run_through
    print0("\n***************************** \n run forward pass during training")
    l0_module.train()
    zs = l0_module.forward(calculate_lagrangian=False)
    for key in zs:
        print0(key, zs[key].shape)

    print0("\n***************************** \n run forward pass during eval")
    l0_module.eval()
    zs = l0_module.forward(calculate_lagrangian=False)
    for key in zs:
        print0(key, zs[key].shape)

    print0("\n***************************** \n run forward pass during lagrangian")
    l0_module.train()
    loss, v = l0_module(calculate_lagrangian=True, pruned_steps=10240)
    print0("loss", loss.item())
    for key in v:
        if torch.is_tensor(v[key]): vv = v[key].item()
        else: vv = v[key]
        print0(key, vv)
    
    print0("\n***************************** \n Test target sparsity") 
    # test target_sparsity
    target_sparsity = l0_module.get_target_sparsity(50, l0_module.target_sparsity)
    print0("target sparsity at step 50: ", target_sparsity)
     
    target_sparsity = l0_module.get_target_sparsity(100, l0_module.target_sparsity)
    print0("target sparsity at step 100: ", target_sparsity)
    
    target_sparsity = l0_module.get_target_sparsity(200, l0_module.target_sparsity)
    print0("target sparsity at step 200: ", target_sparsity)
    
    target_sparsity = l0_module.get_target_sparsity(400, l0_module.target_sparsity)
    print0("target sparsity at step 400: ", target_sparsity)
    
    target_sparsity = l0_module.get_target_sparsity(1000, l0_module.target_sparsity)
    print0("target sparsity at step 1000: ", target_sparsity)

if __name__ == "__main__":
    test_l0_module()
    