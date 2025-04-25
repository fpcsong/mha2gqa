# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team


def get_train_ds_config(offload,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        use_qat=False,
                        max_out_tokens=512):

    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "overlap_comm": False,
        "reduce_bucket_size": "auto",
        "contiguous_gradients": True,
    }
    quantization_config = {
      "enabled": True,
      "quantize_verbose": True,
      "quantizer_kernel": True,
      "quantize-algo": {
        "q_type": "asymmetric"
      },
      "quantize_bits": {
        "start_bits": 16,
        "target_bits": 4
      },
      "quantize_schedule": {
        "quantize_period": 512,
        "schedule_offset": 512
      },
      "quantize_groups": 1,
    }
    return {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "zero_optimization": zero_opt_dict,
        "quantize_training": quantization_config if use_qat else {},
        "bf16":{
            "enabled" : "auto"
        },
        # "postscale_gradients": True,
        "gradient_clipping": "auto",
        "gradient_accumulation_steps": "auto",
        # "scheduler": {
        #     "type": "WarmupCosineLR",
        #     "params": {
        #         "warmup_min_ratio": 0.0,
        #         "warmup_num_steps": 32,
        #         "total_num_steps": "auto"
        #     }
        # },
        # "scheduler": {
        #     "type": "WarmupCosineLR",
        #     "params": {
        #         "warmup_min_ratio": 0,
        #         "warmup_num_steps": "auto",
        #         "total_num_steps": "auto"
        #     }
        # },
        # "optimizer": {
        #     "type": "AdamW",
        #     "params": {
        #     "lr": "auto",
        #     "betas": "auto",
        #     "eps": 1e-8,
        #     "weight_decay": "auto"
        #     }
        # },
    }
