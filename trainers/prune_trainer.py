from config import *
from collections import Counter
from typing import Tuple, Optional, Union, Any
import math
from utils import print0
from distil_losses import *
from transformers import PreTrainedModel, TrainerState
from transformers.cache_utils import DynamicCache
from dataclasses import dataclass
from peft import PeftModel
from trainers.losses import *
from transformers.utils import is_apex_available
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

if is_apex_available():
    from apex import amp

@dataclass
class CustomTrainerState(TrainerState):
    lm_losses: List[float] = None
    lag_losses: List[float] = None
    def __post_init__(self):
        super().__post_init__()
        self.lm_losses = []
        self.lag_losses = []

class CustomTrainerForL0Prune(Seq2SeqTrainer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.step = 0
        self.accumulation_steps = config['accumulation_steps']
        self.lag_lr = config['lag_learning_rate']
        self.acc_lm_losses = []
        self.acc_lag_losses = []
        self.state_lm_losses = []
        self.state_lag_losses = []
    def get_decay_parameter_names(self, model) -> List[str]:
        """
        Get all parameter names that weight decay will be applied to

        Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
        apply to those modules since this function only filter out instance of nn.LayerNorm
        """
        decay_parameters = transformers.trainer_pt_utils.get_parameter_names(
            model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        return decay_parameters
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            if self.args.lr_scheduler_type == 'cosine_with_restarts':
                self.lr_scheduler = transformers.optimization.get_cosine_with_hard_restarts_schedule_with_warmup(
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                    num_cycles=self.args.max_steps // self.args.save_steps,
                    last_epoch=-1
                )
            else:
                self.lr_scheduler = transformers.optimization.get_scheduler(
                    self.args.lr_scheduler_type,
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )
            self._created_lr_scheduler = True
        return self.lr_scheduler
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if transformers.utils.is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {

                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters \
                            and p.requires_grad \
                            and 'l0_module' not in n\
                            and 'kv_layer' not in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters \
                            and p.requires_grad \
                            and 'l0_module' not in n\
                            and 'kv_layer' not in n)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
            ]

            l0_module_params = [p for n, p in opt_model.named_parameters() if "l0_module" in n and "lambda" not in n]
            kv_head_params = [p for n, p in opt_model.named_parameters() if "kv_layer" in n]
            # lagrange_params = [p for n, p in opt_model.named_parameters() if "l0_module" in n and "lambda" in n]
            if len(l0_module_params) > 0:
                optimizer_grouped_parameters.extend([
                    {"params": l0_module_params, "lr": self.lag_lr, "weight_decay": 0.0},
                    ])
            if len(kv_head_params) > 0:
                optimizer_grouped_parameters.extend([
                    {"params": kv_head_params, "lr": self.args.learning_rate * 2, "weight_decay": 0.0},
                    ])
    
            # print0(optimizer_grouped_parameters)
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        num_items_in_batch = None
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            torch.cuda.empty_cache()
        kwargs = {}

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)
            # Finally we need to normalize the loss for reporting
            if num_items_in_batch is None:
                return loss.detach() / self.args.gradient_accumulation_steps
            return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        labels = inputs.get("labels")
        input_ids = inputs.get('input_ids')
        attention_mask = inputs.get('attention_mask')
        weights = inputs.get('weights')
        zs = {key: inputs[key] for key in inputs if "_z" in key}
        prune_steps = self.step // self.accumulation_steps
        if isinstance(model.module, PeftModel):
            target_model = model.module.base_model.model
        else:
            target_model = model.module
        max_prune_steps = target_model.model.l0_module.max_prune_steps
        prune_ratio = prune_steps / max_prune_steps
        self.step += 1 # num forwards
        outputs, zs, lm_loss, lag_loss = target_model.forward_for_train(
            input_ids = input_ids,
            attention_mask=attention_mask,
            pruned_steps=prune_steps,
            **zs
        )
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        length_norm = 'norm' in self.config['loss_type']
        loss_k = 1024 if 'top' in self.config['loss_type'] else -1
        loss_temperature=self.config['temperature'] if 'dyn' in self.config['loss_type'] else 1.0
        lm_loss = fused_loss(
                shift_logits, 
                shift_labels, 
                weights=weights, 
                temperature=loss_temperature, 
                length_norm=length_norm, 
                k=loss_k
        )
        if self.state_lm_losses:
            last_lm_loss = self.state_lm_losses[-1]
            dyn_weights_from_lm = last_lm_loss / (lm_loss.item()+ 1e-6)
            dyn_weights_from_lm = max(0.3, min(3, dyn_weights_from_lm))
        else:
            dyn_weights_from_lm = 1
        if self.state_lag_losses:
            last_lag_loss = self.state_lag_losses[-1]
            dyn_weights_from_lag = ((lag_loss.item() + 1e-6) / (last_lag_loss + 1e-6)) ** 3
            dyn_weights_from_lag = max(0.3, min(3, dyn_weights_from_lag))
        else:
            dyn_weights_from_lag = 1

        lag_loss_weights = dyn_weights_from_lag * dyn_weights_from_lm

        lag_loss_weights = math.exp(prune_steps/max_prune_steps * 3) * lag_loss_weights

        loss = lm_loss + lag_loss * lag_loss_weights
        temp_lm_loss = lm_loss.detach()
        temp_lag_loss = lag_loss.detach()
        dist.all_reduce(temp_lm_loss)
        dist.all_reduce(temp_lag_loss)
        self.acc_lm_losses.append(temp_lm_loss.item() / self.config['world_size'])
        self.acc_lag_losses.append(temp_lag_loss.item() / self.config['world_size'])
        if self.step % self.accumulation_steps == 0 and self.step > 0:
            mean_lm_loss = sum(self.acc_lm_losses) / len(self.acc_lm_losses)
            mean_lag_loss = sum(self.acc_lag_losses) / len(self.acc_lag_losses)
            self.state_lm_losses.append(mean_lm_loss)
            self.state_lag_losses.append(mean_lag_loss)
            self.acc_lm_losses = []
            self.acc_lag_losses = []
            setattr(self.state, 'lm_losses', self.state_lm_losses)
        if prune_steps % 10 == 0 and self.step % self.accumulation_steps == 0:
            with torch.no_grad():
                log_dict = {
                        "prune/lm_loss": self.state_lm_losses[-1],
                        "prune/lag_loss": self.state_lag_losses[-1],
                    }

                for pruning_module in target_model.model.l0_module.pruning_modules:
                    score, sparsity = target_model.model.l0_module.masks[pruning_module].calculate_expected_score_sparsity()
                    mask = target_model.model.l0_module.masks[pruning_module].sample_z(keepdim=True)
                    ts = target_model.model.l0_module.get_relative_target_sparsity(
                        prune_steps,
                        sparsity.detach(),
                        pruning_module
                        )
                    log_dict["sparsity/{}_var".format(pruning_module)] = \
                        mask.squeeze().var(-1, correction=0).mean().item()
                    log_dict["sparsity/{}_mask_mean".format(pruning_module)] = mask.squeeze().mean().item()
                    log_dict["sparsity/{}_mean".format(pruning_module)] = sparsity.mean().item()
                    log_dict["sparsity/{}_target".format(pruning_module)] = ts.mean().item()
                    # if pruning_module in ['hidden']:
                    #     print0('\n', prune_steps, ts, target_model.model.l0_module.masks[pruning_module].target_sparsity)
                self.log(log_dict)

        return (loss, outputs) if return_outputs else loss

class CustomTrainerForDistillationL0Prune(CustomTrainerForL0Prune):
    def __init__(
        self,
        config,
        teacher_model_path,
        temperature=3.0,
        bild_topk=8, 
        **kwargs
        ):
        super().__init__(config, **kwargs)
        self.temperature = temperature
        self.bild_topk=bild_topk
        self.teacher = None
        self.teacher_model_path = teacher_model_path

        self.teacher = AutoModelForCausalLM.from_pretrained(
            self.teacher_model_path, 
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation='eager' if self.config['do_mid_distil'] else 'flash_attention_2'
            )
        if self.teacher:
            if self.is_deepspeed_enabled:
                self.teacher = self._prepare_deepspeed(self.teacher)
                self.teacher.eval()
            else:
                self.teacher = self.accelerator.prepare_model(self.teacher, evaluation_mode=True)
        self.state_distil_losses = []
        self.acc_distil_losses = []
    def _prepare_deepspeed(self, model: PreTrainedModel):
        
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        from copy import deepcopy
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        labels = inputs.get("labels")
        input_ids = inputs.get('input_ids')
        distil_labels = inputs.get('input_ids')
        attention_mask = inputs.get('attention_mask')
        batch_size, seq_len = input_ids.shape
        weights = inputs.get('weights')
        zs = {key: inputs[key] for key in inputs if "_z" in key}
        prune_steps = self.step // self.accumulation_steps
        # if isinstance(model.module, PeftModel):
        #     target_model = model.module.base_model.model
        # else:
        #     target_model = model.module
        target_model = model
        max_prune_steps = target_model.model.l0_module.max_prune_steps
        prune_ratio = prune_steps / max_prune_steps
        self.step += 1 # num forwards
        outputs, zs, lm_loss, lag_loss = target_model.forward_for_train(
            input_ids = input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
            pruned_steps=prune_steps,
            **zs
        )
        logits_s = outputs.logits
        student_hidden_states = outputs.hidden_states
        shift_logits = logits_s[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        if self.teacher.device != model.device:
            self.teacher.to(model.device)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
                output_attentions=True if self.config['do_mid_distil'] else False,
                past_key_values=DynamicCache.from_legacy_cache(None)
            )
        teacher_hidden_states = teacher_outputs.hidden_states

        logits_t =  teacher_outputs.logits.detach()

        logits_s = logits_s[..., :-1, :].contiguous()
        logits_t = logits_t[..., :-1, :].contiguous()
        distil_labels = distil_labels[..., 1:].contiguous()
        attention_mask = attention_mask[..., :-1].contiguous()
        
        # if self.config['lm_loss_weight'] == 0:
        #     logits_t = correct_logits(logits_t, shift_labels)

        with torch.no_grad():
            t_preds = logits_t.argmax(-1)
            s_preds = logits_s.argmax(-1)
            t_correct_pos = ((t_preds == shift_labels) * (shift_labels > -1)).long()
            s_correct_pos = ((s_preds == shift_labels) * (shift_labels > -1)).long()
            correct_pos = t_correct_pos * s_correct_pos
            incorrect_pos = ((1 - correct_pos) * (shift_labels > -1)).long()
        # instruction loss 
        instruction_loss_mask = attention_mask * (shift_labels < 0).long().float()
        # instruction length norm is not a good choice
        # instruction_loss_mask /= (instruction_loss_mask.sum(1, keepdim=True) + 1e-8)
        if weights is not None:
            instruction_loss_mask *= weights.unsqueeze(1)
        instruction_loss_mask = instruction_loss_mask.view(-1)

        origin_instruction_loss = vanilla_kl_loss_func(logits_s, logits_t, temperature=1)
        instruction_loss = (origin_instruction_loss.view(-1) * instruction_loss_mask).sum() / instruction_loss_mask.sum()

        # label loss
        label_loss_mask = (shift_labels > -1).long().float()
        # loss_temperature=self.config['temperature'] if 'dyn' in self.config['loss_type'] else 1.0
        loss_temperature = 1.0 # no dyn for distillation, since confidence is constrained by teacher
        token_loss_weight = evaluate_confidence(logits_t, shift_labels, loss_temperature)

        # skip distil loss if incorrect
        # if self.config['lm_loss_weight'] > 0:
        #     token_loss_weight *= t_correct_pos

        if 'norm' in self.config['loss_type']:
            token_loss_weight /= (label_loss_mask.sum(1, keepdim=True) + 1e-8)
            # token_loss_weight *= (label_loss_mask.sum(1, keepdim=True) + 1e-8) ** (1/5)
        if weights is not None:
            token_loss_weight *= weights.unsqueeze(1)
        token_loss_weight = token_loss_weight.reshape(-1)

        if self.bild_topk > 0:
            t_ld_loss = bild_loss_func(logits_s, logits_t, top_k=self.bild_topk, temperature=self.temperature, student_led=False)
            s_ld_loss = bild_loss_func(logits_s, logits_t, top_k=self.bild_topk, temperature=self.temperature, student_led=True)
            if 'norm' in self.config['loss_type']:
                t_ld_loss = (t_ld_loss.view(-1) * token_loss_weight).sum() / batch_size
                s_ld_loss = (s_ld_loss.view(-1) * token_loss_weight).sum() / batch_size
            else:
                t_ld_loss = (t_ld_loss.view(-1) * token_loss_weight).sum() / label_loss_mask.sum()
                s_ld_loss = (s_ld_loss.view(-1) * token_loss_weight).sum() / label_loss_mask.sum()
            label_kl_loss = t_ld_loss + s_ld_loss
        else:
            label_kl_loss = bild_loss_func_new(logits_s, logits_t, top_k=-self.bild_topk, temperature=self.temperature)
            if 'norm' in self.config['loss_type']:
                label_kl_loss = (label_kl_loss.view(-1) * token_loss_weight).sum() / batch_size
            else:
                label_kl_loss = (label_kl_loss.view(-1) * token_loss_weight).sum() / label_loss_mask.sum()

        distil_loss = instruction_loss + label_kl_loss


        loss = distil_loss + lag_loss
        # print0(lag_loss.item(), dyn_weights_from_lag, dyn_weights_from_distil, lag_loss_weights)

        if self.config['lm_loss_weight'] > 0:
            length_norm = 'norm' in self.config['loss_type']
            loss_k = 1024 if 'top' in self.config['loss_type'] else -1
            loss_temperature=self.config['temperature'] if 'dyn' in self.config['loss_type'] else 1.0
            lm_loss = fused_loss(
                    shift_logits, 
                    shift_labels, 
                    weights=weights, 
                    temperature=loss_temperature, 
                    length_norm=length_norm, 
                    k=loss_k,
                    incorrect_pos=incorrect_pos
            )
        if self.config['lm_loss_weight'] > 0:
            loss += self.config['lm_loss_weight'] * lm_loss

        if lm_loss:
            temp_lm_loss = lm_loss.detach()
        else:
            temp_lm_loss = torch.zeros_like(loss)
        temp_lag_loss = lag_loss.detach()
        temp_distil_loss = distil_loss.detach()
        dist.all_reduce(temp_distil_loss)
        dist.all_reduce(temp_lm_loss)
        dist.all_reduce(temp_lag_loss)
        self.acc_lm_losses.append(temp_lm_loss.item() / self.config['world_size'])
        self.acc_lag_losses.append(temp_lag_loss.item() / self.config['world_size'])
        self.acc_distil_losses.append(temp_distil_loss.item() / self.config['world_size'])
        if self.step % self.accumulation_steps == 0 and self.step > 0:
            mean_lm_loss = sum(self.acc_lm_losses) / len(self.acc_lm_losses)
            mean_lag_loss = sum(self.acc_lag_losses) / len(self.acc_lag_losses)
            mean_distil_loss = sum(self.acc_distil_losses) / len(self.acc_distil_losses)
            self.state_lm_losses.append(mean_lm_loss)
            self.state_lag_losses.append(mean_lag_loss)
            self.state_distil_losses.append(mean_distil_loss)
            self.acc_lm_losses = []
            self.acc_lag_losses = []
            self.acc_distil_losses = []
            setattr(self.state, 'lm_losses', self.state_lm_losses)
        if prune_steps % 10 == 0 and self.step % self.accumulation_steps == 0:
            with torch.no_grad():
                log_dict = {
                        "prune/lm_loss": self.state_lm_losses[-1],
                        "prune/lag_loss": self.state_lag_losses[-1],
                        "prune/distil_loss": self.state_distil_losses[-1],
                    }

                for pruning_module in target_model.model.l0_module.pruning_modules:
                    score, sparsity = target_model.model.l0_module.masks[pruning_module].calculate_expected_score_sparsity()
                    mask = target_model.model.l0_module.masks[pruning_module].sample_z(keepdim=True)
                    ts = target_model.model.l0_module.get_relative_target_sparsity(
                        prune_steps,
                        sparsity.detach(),
                        pruning_module
                        )
                    log_dict["sparsity/{}_var".format(pruning_module)] = \
                        mask.squeeze().var(-1, correction=0).mean().item()
                    log_dict["sparsity/{}_mask_mean".format(pruning_module)] = mask.squeeze().mean().item()
                    log_dict["sparsity/{}_mean".format(pruning_module)] = sparsity.mean().item()
                    log_dict["sparsity/{}_target".format(pruning_module)] = ts.mean().item()

                self.log(log_dict)

        return (loss, outputs) if return_outputs else loss