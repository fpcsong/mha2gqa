from config import *
from collections import Counter
from typing import Tuple, Optional, Union
import math
from utils import print0
from distil_losses import *
from transformers import PreTrainedModel
from transformers.cache_utils import DynamicCache
from trainers.losses import *

class CustomTrainerForSFT(Seq2SeqTrainer):
    def __init__(self, config, **kwargs):
        self.config = config
        super().__init__(**kwargs)
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
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        input_ids = inputs.get('input_ids')
        attention_mask = inputs.get('attention_mask')
        weights = inputs.get('weights')
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.get("logits")

        # Shift so that tokens < n predict n
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
        return (lm_loss, outputs) if return_outputs else lm_loss

class CustomTrainerForDistillation(CustomTrainerForSFT):
    def __init__(
        self, 
        config, 
        teacher_model_path, 
        temperature=3.0, 
        bild_topk=8,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.config = config
        self.temperature = temperature
        self.bild_topk = bild_topk
        self.teacher_model_path = teacher_model_path
        # self.teacher = None
        self.teacher = AutoModelForCausalLM.from_pretrained(
            self.teacher_model_path, 
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation='flash_attention_2'
            )
        if self.teacher:
            if self.is_deepspeed_enabled:
                self.teacher = self._prepare_deepspeed(self.teacher)
                self.teacher.eval()
            else:
                self.teacher = self.accelerator.prepare_model(self.teacher, evaluation_mode=True)

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
        input_ids = inputs.get('input_ids')
        labels = inputs.get("labels")
        attention_mask = inputs.get('attention_mask')
        batch_size, seq_len = input_ids.shape

        shift_labels = labels[..., 1:].contiguous()
        weights = inputs.get('weights')

        student_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits_s = student_outputs.logits.contiguous()
        shift_logits = logits_s[..., :-1, :].contiguous()

        if self.teacher.device != model.device:
            self.teacher.to(model.device)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                past_key_values=DynamicCache.from_legacy_cache(None)
            )
        logits_t =  teacher_outputs['logits'].detach().contiguous()

        logits_s = logits_s[..., :-1, :].contiguous()
        logits_t = logits_t[..., :-1, :].contiguous()
        attention_mask = attention_mask[..., :-1].contiguous()
        label_loss_mask = (shift_labels > -1).long().float()
        
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

        loss = distil_loss
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
        else:
            lm_loss = 0
        if self.config['lm_loss_weight'] > 0:
            loss += self.config['lm_loss_weight'] * lm_loss

        return (loss, student_outputs) if return_outputs else loss