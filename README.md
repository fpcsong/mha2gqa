## 0. Must Read
1. This code is applicable to standard models with the Llama 2 architecture (DeepSeek V1 shares the same architecture as Llama 2).
2. All experimental data used in this study have been included in this repository. No need to download again.

## 1. Environment

```bash
pip install -r requirements.txt
```


## 2. Prepare your model and calibration data
download llama-7B or Sheared-llama-1.3B
```bash
git clone https://huggingface.co/princeton-nlp/Sheared-LLaMA-1.3B
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf
```

This command calibrates the model using the C4 dataset, with a total token count of 262144 (128*2048). Additionally, it saves the key and value caches for each layer of the model in <save_path>.
```bash
python MHA2GQA/get_calibration.py --model_path <base model path> --save_path <the folder to save calibration data> --batch_size <int, depends on your cuda memory> --device <cuda:0 for example>
```


## 3. Experiments

### 3.1 SFT teacher
```bash
#train llama-7B teacher (train steps equal to 2 epoch)
bash shells/sft-llama.sh --base_model_path <llama-base-7B path> --train_steps 2306 --batch_size 128 --experiment_name teacher --num_gpus 8
#train Sheared-llama-1.3B teacher (train steps equal to 3 epoch)
bash shells/sft-llama.sh --base_model_path <Sheared-llama-1.3B path> --train_steps 3459 --batch_size 128 --experiment_name teacher --num_gpus 8
#evaluate trained teacher model
bash shells/vllm-eval-llama.sh --checkpoint_path <checkpoint path>
```

### 3.2 train GQA baseline
```bash
# train llama-7B GQA-16 baseline (train steps equal to 5 epoch)
bash shells/distill0prune-llama.sh --base_model_path <llama-base-7B path> --prune_config ./prune_utils/config/llama2-7b-gqa-16groups.yaml --teacher_path <SFT teacher path> --train_steps 11530 --batch_size 64 --experiment_name group16 --num_gpus 8

# train llama-7B GQA-8 baseline (train steps equal to 10 epoch)
bash shells/distill0prune-llama.sh --base_model_path <llama-base-7B path> --prune_config ./prune_utils/config/llama2-7b-gqa-8groups.yaml --teacher_path <SFT teacher path> --train_steps 23060 --batch_size 64 --experiment_name group8 --num_gpus 8

# train llama-7B GQA-4 baseline (train steps equal to 15 epoch)
bash shells/distill0prune-llama.sh --base_model_path <llama-base-7B path> --prune_config ./prune_utils/config/llama2-7b-gqa-4groups.yaml --teacher_path <SFT teacher path> --train_steps 34590 --batch_size 64 --experiment_name group4 --num_gpus 8

# train Sheared-llama-1.3B GQA-8 baseline (train steps equal to 6 epoch)
bash shells/distill0prune-llama.sh --base_model_path <Sheared-llama-1.3B path> --prune_config ./prune_utils/config/llama2-1.3B-gqa-8groups.yaml --teacher_path <SFT teacher path> --train_steps 13836 --batch_size 64 --experiment_name group8 --num_gpus 8

# train Sheared-llama-1.3B GQA-4 baseline (train steps equal to 12 epoch)
bash shells/distill0prune-llama.sh --base_model_path <Sheared-llama-1.3B path> --prune_config ./prune_utils/config/llama2-1.3B-gqa-4groups.yaml --teacher_path <SFT teacher path> --train_steps 27672 --batch_size 64 --experiment_name group4 --num_gpus 8
```

### 3.3 train transformed GQA
```bash
# transform llama-7B model and prune it to GQA-16 (train steps equal to 5 epoch)
bash shells/transform-distill0prune-llama.sh --calibration_data_path <the folder to save calibration data> --base_model_path <llama-base-7B path> --save_model_path <the folder to save transformed model> --prune_config ./prune_utils/config/llama2-7b-gqa-16groups.yaml --teacher_path <SFT teacher path> --train_steps 11530 --batch_size 64 --experiment_name group16 --num_gpus 8 --group_num 16 --group_criterion <dist/cos> --item <value/key/none>

# transform llama-7B model and prune it to GQA-8 (train steps equal to 10 epoch)
bash shells/transform-distill0prune-llama.sh --calibration_data_path <the folder to save calibration data> --base_model_path <llama-base-7B path> --save_model_path <the folder to save transformed model> --prune_config ./prune_utils/config/llama2-7b-gqa-8groups.yaml --teacher_path <SFT teacher path> --train_steps 23060 --batch_size 64 --experiment_name group8 --num_gpus 8 --group_num 8 --group_criterion <dist/cos> --item <value/key/none>

# transform llama-7B model and prune it to GQA-4 (train steps equal to 15 epoch)
bash shells/transform-distill0prune-llama.sh --calibration_data_path <the folder to save calibration data> --base_model_path <llama-base-7B path> --save_model_path <the folder to save transformed model> --prune_config ./prune_utils/config/llama2-7b-gqa-4groups.yaml --teacher_path <SFT teacher path> --train_steps 34590 --batch_size 64 --experiment_name group4 --num_gpus 8 --group_num 4 --group_criterion <dist/cos> --item <value/key/none>

# transform Sheared-llama-1.3B and prune it to GQA-8 (train steps equal to 6 epoch)
bash shells/transform-distill0prune-llama.sh --calibration_data_path <the folder to save calibration data> --base_model_path <Sheared-llama-1.3B path> --save_model_path <the folder to save transformed model> --prune_config ./prune_utils/config/llama2-1.3B-gqa-8groups.yaml --teacher_path <SFT teacher path> --train_steps 13836 --batch_size 64 --experiment_name group8 --num_gpus 8 --group_num 8 --group_criterion <dist/cos> --item <value/key/none>

# transform Sheared-llama-1.3B and prune it to GQA-4 (train steps equal to 12 epoch)
bash shells/transform-distill0prune-llama.sh --calibration_data_path <the folder to save calibration data> --base_model_path <Sheared-llama-1.3B path> --save_model_path <the folder to save transformed model> --prune_config ./prune_utils/config/llama2-1.3B-gqa-8groups.yaml --teacher_path <SFT teacher path> --train_steps 27672 --batch_size 64 --experiment_name group4 --num_gpus 8 --group_num 4 --group_criterion <dist/cos> --item <value/key/none>
```

