from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import os
import time
from utils import moving_average, print0, print_in_rank
import torch
import copy
import torch
from deepspeed.runtime.zero import GatheredParameters
import deepspeed

class ExponentialMovingAverage:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'l0_module' not in name:
            # if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name].to(param.data.device)
                self.shadow[name] = new_average.clone()
    def online_update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name].to(param.data.device)
                self.shadow[name] = new_average
                param.data = new_average
    def keep_change(self):
        self.backup = copy.deepcopy(self.shadow)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data
                param.data = self.shadow[name].to(param.data.device)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class EMACallback(TrainerCallback):
    def __init__(self, model, decay=0.999):
        self.ema = ExponentialMovingAverage(model, decay)
        self.ema.register()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step > 0:
            # check loss
            curr_lm_loss = state.lm_losses[-1]
            mean_recent_lm_loss = state.lm_losses[-10:]
            threshold = sum(mean_recent_lm_loss) / len(mean_recent_lm_loss)
            if curr_lm_loss < threshold * 1.5 or curr_lm_loss < 0.1:
                self.ema.decay = 0.8
            else:
                self.ema.decay = 0.99
            self.ema.online_update()
    # def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     self.ema.apply_shadow()

class SparsityCallback(TrainerCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step > 0:
            kwargs['model'].model.l0_module.constrain_sparsity(state.global_step)
        return control

class PeftCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        # peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(checkpoint_folder)
        kwargs["model"].config.save_pretrained(checkpoint_folder)
        kwargs["tokenizer"].save_pretrained(checkpoint_folder)
        # pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        # if os.path.exists(pytorch_model_path):
        #     os.remove(pytorch_model_path)

        return control

class DistilCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "pytorch_model")
        kwargs["model"].save_pretrained(peft_model_path)

        return control


class ProcessBarCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.start_time = -1
                
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.start_time < 0:
            self.start_time = time.time()
        if state.global_step % 10 == 0 and state.is_world_process_zero:
            used_time_sec = time.time()-self.start_time
            eta_sec = used_time_sec / state.global_step * state.max_steps - used_time_sec
            used_time_str = '{}:{}:{}'.format(
                int(used_time_sec//3600), 
                int((used_time_sec%3600)//60), 
                int(used_time_sec%60)
                )
            eta_time_str = '{}:{}:{}'.format(
                int(eta_sec//3600), 
                int((eta_sec%3600)//60), 
                int(eta_sec%60)
                )
            print('\n{}/{}, {}/{}'.format(state.global_step, state.max_steps, used_time_str, eta_time_str))
        return super().on_step_end(args, state, control, **kwargs)

class PruneCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.model_ema = None

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model'].model.l0_module
        
        if not model.training:
            return super().on_step_begin(args, state, control, **kwargs)
        
        model.constrain_sparsity(state.global_step)
        # 存储ema状态
        if self.model_ema is None:
            self.model_ema = copy.deepcopy(model)
        with torch.no_grad():
            for src_param, dst_param in zip(model.parameters(), self.model_ema.parameters()):
                dst_param.data.copy_(src_param.data)

        return super().on_step_begin(args, state, control, **kwargs)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model'].model.l0_module
        if not model.training:
            return super().on_step_end(args, state, control, **kwargs)
        name2alpha = {}
        warmup_ratio = min(1, state.global_step / model.lagrangian_warmup_steps)
        for module in model.pruning_modules:
            curr_target_sparsity = model.masks[module].target_sparsity * warmup_ratio
            name2alpha[module] = abs(curr_target_sparsity - model.pre_sparsity[module].mean().item())

        base_alpha = max(0.002, min(1, (state.global_step / model.max_prune_steps)) ** 5 / 2)
        for (src_name, src_param), (dst_name, dst_param) in zip(model.named_parameters(), self.model_ema.named_parameters()):
            if src_param.nelement() == 1:
                continue
            for module in model.pruning_modules:
                if module in src_name:
                    alpha = min(1, max(base_alpha, name2alpha[module]))
                    # print0('\n module, alpha, curr, tgt: ', module, alpha, model.pre_sparsity[module].mean().item(), model.masks[module].target_sparsity * warmup_ratio)
                    break
            # print0('\nafter step before merge')
            # sync deepspeed-> transformer
            src_param.data = deepspeed.utils.safe_get_full_fp32_param(src_param).data.clone()
            # print0(src_param.data[0])
            # merge in transformer
            src_param.data = dst_param.data.clone() * alpha + src_param.data.clone() * (1-alpha)
            # sync transformers -> deepspeed
            deepspeed.utils.safe_set_full_fp32_param(src_param, src_param.data)
            # check
            # src_param.data = deepspeed.utils.safe_get_full_fp32_param(src_param).data.clone()
            # print0('\nafter step after merge')
            # print0(deepspeed.utils.safe_get_full_fp32_param(src_param).data[0])
            # print0(src_param.data[0])
        return super().on_step_end(args, state, control, **kwargs)

    def _on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model'].model.l0_module
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, norm_type=2.0)
        return super().on_substep_end(args, state, control, **kwargs)