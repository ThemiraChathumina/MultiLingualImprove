from transformers import AutoModelForCausalLM, AutoModel, AutoConfig
import torch
from transformers import AutoTokenizer
from torch import nn
import torch.nn.functional as F
import csv 
import numpy as np
from Configs import IDPConfigs

class JSMMultilingualEmbeddingModel(nn.Module):
    ALLOWED_MODELS = {
        "FacebookAI/xlm-roberta-base",
        "FacebookAI/xlm-roberta-large",
        "facebook/nllb-200-distilled-600M",
        "facebook/nllb-200-1.3B",
        "google/mt5-small",
        "google/mt5-base",
        "google/mt5-large",
        "google/mt5-xl",
        "DKYoon/mt5-small-lm-adapt",
        "DKYoon/mt5-large-lm-adapt",
        "DKYoon/mt5-xl-lm-adapt",
        "facebook/nllb-200-distilled-1.3B"
    }
    
    def __init__(self, embedding_model_base, embedding_model_ext, max_seq_len, freeze_embedding = True):
        super().__init__()

        if embedding_model_base not in self.ALLOWED_MODELS:
            raise ValueError(f"Model is not in allowed models: {self.ALLOWED_MODELS}")
        
        self.embedding_model_base = AutoModel.from_pretrained(embedding_model_base)
        if "nllb" in embedding_model_base or "mt5" in embedding_model_base:
            self.embedding_model_base = self.embedding_model_base.encoder 

        self.freeze_embedding = freeze_embedding
        if freeze_embedding:
            for param in self.embedding_model_base.parameters():
                param.requires_grad = False
            
        self.tokenizer_base = AutoTokenizer.from_pretrained(embedding_model_base)

        self.max_seq_len = max_seq_len
        
        self.embedding_dim_base = self.embedding_model_base.config.hidden_size
        self.embedding_dim = self.embedding_dim_base

        num_embedding_tokens = 1
        self.num_embedding_tokens = num_embedding_tokens
        self.learnable_queries_base = None
        
        # If using prepended queries, initialize them.
        
       
        self.configs = IDPConfigs()

        self.langbridge_baseline = self.configs.baseline

        if not self.langbridge_baseline:
            self.num_layers = self.embedding_model_base.config.num_hidden_layers  # Exclude embedding layer
            self.layer_weights_lb = nn.Parameter(torch.full((self.num_layers,), 3e-5))

            self.base_temp = torch.tensor(1e2)
            self.factor = torch.tensor(1e5)
            self.temp = nn.Parameter(torch.tensor(1e-5))
        
        self.csv_file = open(self.configs.gate_csv_path, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        
    def get_input_embeddings(self, model, input_ids):
        if "M2M" in model.__class__.__name__:
            return model.embed_tokens(input_ids)
        return model.get_input_embeddings()(input_ids)

    def mt_input_features(self, input_texts_m2m):
        input_ids_m2m, attention_mask_m2m = [], []
        for input_text_m2m in input_texts_m2m:
            encoding_m2m = self.tokenizer_base(input_text_m2m,
                                         padding='longest',
                                         max_length=self.max_seq_len,
                                         truncation=True)
            input_ids_m2m_item = encoding_m2m.input_ids
            attention_mask_m2m_item = encoding_m2m.attention_mask
            input_ids_m2m.append(input_ids_m2m_item)
            attention_mask_m2m.append(attention_mask_m2m_item)
        max_len = max([len(item) for item in input_ids_m2m])
        m2m_pad_id = self.tokenizer_base.pad_token_id
        for input_ids_m2m_item, attention_mask_m2m_item in zip(input_ids_m2m, attention_mask_m2m):
            while len(input_ids_m2m_item) < max_len:
                input_ids_m2m_item.append(m2m_pad_id)
                attention_mask_m2m_item.append(0)
        input_ids_m2m = torch.tensor(input_ids_m2m, dtype=torch.long).cuda()
        attention_mask_m2m = torch.tensor(attention_mask_m2m, dtype=torch.long).cuda()
        return input_ids_m2m, attention_mask_m2m
    
    def get_last_hidden_states(self, encoded_inputs):
        input_ids, attention_mask = self.mt_input_features(encoded_inputs)
        outputs = self.embedding_model_base(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state, attention_mask
    
    def softmax_gated(self, encoded_inputs):
        input_ids, attention_mask = self.mt_input_features(encoded_inputs)

        outputs = self.embedding_model_base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Skip embeddings layer (hidden_states[0]), only use actual transformer layers
        hidden_states_stacked = torch.stack(outputs.hidden_states[1:], dim=1)  # [B, L, T, D]

        temp_applicable = self.base_temp + self.temp * self.factor
        norm_weights = F.softmax(temp_applicable * self.layer_weights_lb, dim=0)  # [L]
        norm_weights = norm_weights.view(1, -1, 1, 1)

        fused_tokens = torch.sum(hidden_states_stacked * norm_weights, dim=1)  # [B, T, D]
        return fused_tokens, attention_mask

        
    
    def forward(self, encoded_inputs):
        if not self.langbridge_baseline:
            base_embeddings, base_attention_mask = self.softmax_gated(
                encoded_inputs
            )
        else:
            base_embeddings, base_attention_mask = self.get_last_hidden_states(
                encoded_inputs
            )
        return base_embeddings, base_attention_mask
        
    
    def log_softmax_gate(self):
        if not self.langbridge_baseline:
            with torch.no_grad():
                temp_applicable = self.base_temp + self.temp * self.factor
                softmaxed_weights = F.softmax(temp_applicable * self.layer_weights_lb, dim=0).detach().to(torch.float32).cpu().numpy()
                np.round(softmaxed_weights, decimals=4)
                self.csv_writer.writerow(softmaxed_weights.flatten().tolist() + [temp_applicable.item()])
            
    def print_softmax_gate(self):
        print("Softmax Gate Weights:")
        print(self.layer_weights_lb.data.cpu().numpy())
        print("Softmax Gate")
        print(F.softmax(self.temp * self.layer_weights_lb, dim=0).data.cpu().numpy())

    def close_csv(self):
        if self.csv_file:
            self.csv_file.close()