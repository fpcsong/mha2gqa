import argparse
from datasets import load_dataset
from torch import nn
from typing import Optional, Tuple
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers import LlamaConfig, LlamaModel,LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer,LlamaAttention, LlamaRMSNorm, LlamaRotaryEmbedding
import torch
import os
from torch.utils.data import DataLoader

class CustomLlamaAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int,save_path:str,calibration_max_length:int):
        super().__init__(config, layer_idx)
        self.save_path=save_path
        self.calibration_max_length=calibration_max_length
        self.key_data=None
        self.value_data=None
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if (self.key_data is None) or self.key_data.shape[0]<self.calibration_max_length:
            if self.key_data is not None:
                self.key_data=torch.cat((self.key_data, key_states.to('cpu')), dim=0)
                self.value_data=torch.cat((self.value_data, value_states.to('cpu')), dim=0)
            else:
                self.key_data=key_states.to('cpu')
                self.value_data=value_states.to('cpu')
            print("layer"+str(self.layer_idx)+" length"+str(self.key_data.shape[0]))

            if self.key_data.shape[0]>=self.calibration_max_length:
                head_num=hidden_states.shape[-1]//self.head_dim
                self.key_data=self.key_data[:self.calibration_max_length]
                self.value_data=self.value_data[:self.calibration_max_length]

                self.key_data=self.key_data.transpose(0,1).reshape(head_num,-1,self.head_dim)
                self.value_data=self.value_data.transpose(0,1).reshape(head_num,-1,self.head_dim)

                torch.save(self.key_data, self.save_path+'/layer'+str(self.layer_idx)+'-key.pt')
                torch.save(self.value_data, self.save_path+'/layer'+str(self.layer_idx)+'-value.pt')
        
        return super().forward(hidden_states,
            position_embeddings,
            attention_mask,
            past_key_value,
            cache_position,
            **kwargs
        )  

class CustomDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int,save_path:str,calibration_max_length:int):
        super().__init__(config,layer_idx)
        self.self_attn = CustomLlamaAttention(config=config, layer_idx=layer_idx, save_path=save_path, calibration_max_length=calibration_max_length)

class CustomLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig,save_path:str,calibration_max_length:int):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [CustomDecoderLayer(config, layer_idx,save_path,calibration_max_length) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

def main(args):
    os.makedirs(args.save_path, exist_ok=True)
    
    tokenizer=LlamaTokenizer.from_pretrained(args.model_path)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    tokenizer.unk_token_id = 0

    tokenizer.add_special_tokens = False

    model_dtype = torch.bfloat16 if args.bf16 else torch.float16

    model = CustomLlamaModel.from_pretrained(
                            args.model_path,
                            torch_dtype=torch.bfloat16,
                            device_map=None, 
                            save_path=args.save_path,
                            calibration_max_length=args.calibration_max_length
                        )
    
    model=model.to(args.device)
    model=model.to(model_dtype)

    
    def filter_long_text(example):

        tokenized_length = len(tokenizer(example["text"], truncation=False)["input_ids"])

        return tokenized_length >= 2048

    
    dataset = load_dataset("json", data_files=args.dataset_path,split='train')
    dataset = dataset.filter(filter_long_text)

    dataset = dataset.select(range(128))


    texts = [example["text"] for example in dataset]
    dataloader = DataLoader(texts,batch_size=args.batch_size)
    for batch_texts in dataloader:
        inputs = tokenizer(
            batch_texts,
            max_length=2048,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to(args.device)  


        with torch.no_grad():
            outputs = model(**inputs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calibrate a model')

    parser.add_argument('-bf16', '--bf16', type=bool,default=True)
    parser.add_argument('--model_path', type=str,default='/workspace/Sheared-LLaMA-1.3B')
    parser.add_argument('--dataset_path', type=str,default='benchmarking/c4-train.00000-of-01024.jsonl')
    parser.add_argument('--save_path', type=str,default='/workspace/calibration_data')   
    parser.add_argument('--batch_size', type=int,default=8)
    parser.add_argument('--device', type=str,default='cuda:0')
    parser.add_argument('--calibration_max_length', type=int,default=128)
    
    args = parser.parse_args()

    main(args)