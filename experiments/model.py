from transformers import AutoModelForCausalLM, AutoModel, AutoConfig
import torch
from transformers import AutoTokenizer
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModel, AutoConfig
import torch
from transformers import AutoTokenizer
from torch import nn
import torch.nn.functional as F
import csv 
import numpy as np

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
        
        self.num_layers = self.embedding_model_base.config.num_hidden_layers  # Exclude embedding layer
        self.layer_weights_lb = nn.Parameter(torch.full((self.num_layers,), 3e-5))

        self.base_temp = torch.tensor(1e2)
        self.factor = torch.tensor(1e5)
        self.temp = nn.Parameter(torch.tensor(1e-5))
        

        
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
        base_embeddings, base_attention_mask = self.softmax_gated(
                encoded_inputs
            )
        return base_embeddings, base_attention_mask
        

class MLP(nn.Module):
    def __init__(self, mt_dim, llm_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(mt_dim, mt_dim * 2)
        self.linear2 = nn.Linear(mt_dim * 2, llm_dim)
        self.relu = nn.ReLU()
    def forward(self, mt_hidden_state):
        output = self.linear1(mt_hidden_state)
        output = self.relu(output)
        output = self.linear2(output)
        return output

class Mapper(nn.Module):
    def __init__(self, input_size, output_size):
        super(Mapper, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class LinearMapper(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearMapper, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)
    
class Mapping(nn.Module):
    def __init__(self, mt_dim, llm_dim):
        super(Mapping, self).__init__()
        self.mlp = LinearMapper(mt_dim, llm_dim)
        self.end_boundary = nn.Parameter(
            torch.zeros(1, 1, llm_dim), requires_grad=True
        )
    def forward(self, hidden_states):
        hidden_states = self.mlp(hidden_states)
        return hidden_states

    def get_embed(self):
        return self.end_boundary

    
class MPTModel(nn.Module):
    def __init__(self, config):
        super(MPTModel, self).__init__()
        self.config = config  # Ensure there is a config attribute
        self.max_gen_len = config['max_gen_len']

        self.encoder_mt = JSMMultilingualEmbeddingModel(config['mt_path'], config['ext_path'], config['max_seq_len'])

        model_llm = AutoModelForCausalLM.from_pretrained(config['llm_path'])

        self.model_llm = model_llm

        self.llm_embedding_layer = self.model_llm.get_input_embeddings()
        for name, parameter in self.model_llm.named_parameters():
            parameter.requires_grad = False

        d_model = self.encoder_mt.embedding_dim
        self.mapping = Mapping(d_model, model_llm.config.hidden_size)
        self.llm_pad_token_id = config['llm_pad_token_id']
        self.llm_bos_token_id = config['llm_bos_token_id']
        print('mapping layer size:', sum(param.numel() for param in self.mapping.parameters()) / 1000000)

    def squeeze_pad(self, hidden_states, masks):
        x_01 = (masks != 0).long()

        seq_num_len = x_01.size(1)
        offset = torch.tensor([(i + 1) for i in range(seq_num_len)], dtype=torch.long).to(x_01.device)
        offset = offset.unsqueeze(dim=0).expand_as(x_01)
        x_01 *= offset
        _, idx = x_01.sort(1, descending=False)

        masks = masks.gather(1, idx)
        idx = idx.unsqueeze(dim=-1).expand_as(hidden_states)
        hidden_states = hidden_states.gather(1, idx)

        bs, seq_len, dim = hidden_states.size()
        masks_sum = torch.sum(masks, dim=0)
        idx = masks_sum > 0
        idx = idx.unsqueeze(dim=0).expand_as(masks)
        masks = masks[idx]
        idx_ex = idx.unsqueeze(dim=-1).expand_as(hidden_states)
        hidden_states = hidden_states[idx_ex]
        hidden_states = hidden_states.view(bs, -1, dim)
        masks = masks.view(bs, -1)

        return hidden_states, masks, idx

    def forward(self, encoded_inputs,
                labels=None, mask_label=None, input_ids_prompt=None, mask_prompt=None):
        end_boundary = self.mapping.get_embed()
        bs = len(encoded_inputs)
        end_boundary = end_boundary.expand([bs, 1, end_boundary.size(-1)])

        bos = torch.tensor([self.llm_bos_token_id for i in range(bs)], dtype=torch.long).cuda()
        bos_embedding = self.llm_embedding_layer(bos)
        bos_embedding = bos_embedding.view(bs, 1, -1)
        mask = torch.ones([bs, 1], dtype=torch.long).cuda()
        llm_input_embedding = bos_embedding
        llm_input_mask = mask

        mt_encoder_outputs, attention_mask_mt = self.encoder_mt(encoded_inputs)
        
        mt_hidden_state = self.mapping(mt_encoder_outputs)
        llm_input_embedding = torch.cat([llm_input_embedding, mt_hidden_state, end_boundary],
                                        dim=1)
        llm_input_mask = torch.cat([llm_input_mask, attention_mask_mt, mask], dim=1)

        if input_ids_prompt is not None:

            hidden_states_prompt = self.llm_embedding_layer(input_ids_prompt)
            llm_input_embedding = torch.cat([llm_input_embedding, hidden_states_prompt], dim=1)
            llm_input_mask = torch.cat([llm_input_mask, mask_prompt], dim=1)
        if labels is not None:
            pad_labels = llm_input_mask * -100 + (1 - llm_input_mask) * -100
            label_embedding = self.llm_embedding_layer(labels)
            llm_input_embedding = torch.cat([llm_input_embedding, label_embedding], dim=1)
            llm_input_mask = torch.cat([llm_input_mask, mask_label], dim=1)
            labels = labels * mask_label - 100 * (1 - mask_label)
            labels = torch.cat([pad_labels, labels], dim=1)

        llm_input_embedding, llm_input_mask, cut_pad_idx \
            = self.squeeze_pad(llm_input_embedding, llm_input_mask)

        if labels is None:
            generate_ids = self.model_llm.generate(inputs_embeds=llm_input_embedding,
                                                   attention_mask=llm_input_mask,
                                                   max_new_tokens=self.max_gen_len,
                                                   pad_token_id=self.llm_pad_token_id,
                                                   do_sample=False)
            return generate_ids
        else:
            bs, seq_len = labels.size()
            labels = labels[cut_pad_idx]
            labels = labels.view(bs, -1)
            output = self.model_llm(inputs_embeds=llm_input_embedding,
                                    attention_mask=llm_input_mask,
                                    labels=labels)
            return output.loss
    