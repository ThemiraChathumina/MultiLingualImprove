# Copyright 2023-present the HuggingFace Inc. team.
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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TRANSFORMERS_MODEL_CONFIG


class AdaptedAttention(nn.Module):
    """This module wraps a LLamaAttention module and injects adaption prompts."""

    def __init__(self, model_type: str, model):
        """
        Initialize object.

        Args:
            model_type: The transformer model type. This is used to retrieve the right method to
                compute query states.
            adapter_len: The length of the adaption prompt to insert.
            model: The original transformer attention module that is being wrapped.
        """
        assert not isinstance(model, AdaptedAttention)
        super().__init__()
        self.model_type = model_type
        self.model = model
        #self.adapter_states = adapter_states
        #self.prefix_vec = prefix_vec
        # Assume all parameters of the attention model we are wrapping are on the same device.
        device = next(model.parameters()).device
        # Don't think this was specified in the paper, but we follow the official repo which used an Embedding
        # which initializes the tokens with standard normal values.
        # https://github.com/ZrrSkywalker/LLaMA-Adapter/blob/41c3546fe1997ab8a65809dc8d8f9252b19d9faf/llama/model.py#L234
        # (bsz, adapter_len, hidden_size)
        target_dtype = (
            model.q_proj.weight.dtype if model.q_proj.weight.dtype not in [torch.int8, torch.uint8] else torch.float32
        )
        self.adapter_states =  torch.empty(1)
        # self.adaption_prompt_k = prefix_states[layer_index, 0, :, :, :, :]
        # self.adaption_prompt_v = prefix_states[layer_index, 1, :, :, :, :]
        # Initialize the gate to 0 as this is "zero-init".
        self.adaption_gate = nn.Parameter(torch.zeros(1))
        # self.adaption_prompt = nn.Parameter(
        #     torch.empty(1, 1, self.model.hidden_size)
        # )

    def update_adapter_states(self, adapter_states):
        self.adapter_states = adapter_states

    def forward(self, **kwargs):
        """
        Forward pass for the adapter which wraps the original LlamaAttention module.

        "Official" paper implementation:
        https://github.com/ZrrSkywalker/LLaMA-Adapter/blob/41c3546fe1997ab8a65809dc8d8f9252b19d9faf/llama/model.py#L141

        Args:
            kwargs: See the original LlamaAttention module.
        """
        if kwargs.get("output_attention", False):
            raise NotImplementedError("output_attention is not currently supported.")

        output, _, past_key_value = self.model(**kwargs)
        bsz = output.shape[0]
        q_len = output.shape[1]
        #embed_dim = output.shape[2]
        prefix_len = self.adapter_states[0].shape[1]
        k_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].k_proj_layer
        v_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].v_proj_layer
        o_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].o_proj_layer
        factor = (
            self.model.k_proj.in_features // self.model.k_proj.out_features
        )  # Mistral has different input and output dimension for k_proj and v_proj layers

        # if k_proj_layer == v_proj_layer:
        #     _, key, value = getattr(self.model, k_proj_layer)(self.adapter_states).split(embed_dim, dim=2)
        # else:
        #self.adaption_prompt = nn.Parameter(self.adaption_prompt.repeat(batch_size, 1, 1))
        #self.adaption_prompt = self.adaption_prompt.expand(bsz, 1, self.model.hidden_size)
        adapter_states_k = self.adapter_states[0]
        adapter_states_v = self.adapter_states[1]
        #adapter_states_k = torch.cat((self.adaption_prompt, adapter_states_k), dim=1)
        #adapter_states_v = torch.cat((self.adaption_prompt, adapter_states_v), dim=1)
        key = getattr(self.model, k_proj_layer)(adapter_states_k)
        value = getattr(self.model, v_proj_layer)(adapter_states_v)
        
        # (bsz, nadapter_len, dim)
        adapter_k = (
            key.view(bsz, prefix_len, (self.model.num_heads // factor), self.model.head_dim)
            .transpose(1, 2)
        )
        adapter_v = (
            value.view(bsz, prefix_len, (self.model.num_heads // factor), self.model.head_dim)
            .transpose(1, 2)
        )
        # Below is taken from https://github.com/huggingface/transformers/blob/e547458c43dfdbbb8f6a7757237e234c44e20a8f/src/transformers/models/mistral/modeling_mistral.py#L181
        # (bsz, num_heads, adapter_len, head_dim)
        adapter_k = torch.repeat_interleave(adapter_k, repeats=factor, dim=1)
        adapter_v = torch.repeat_interleave(adapter_v, repeats=factor, dim=1)
        # Recompute query states.
        compute_query_states = TRANSFORMERS_MODEL_CONFIG[self.model_type].compute_query_states
        # (bsz, num_heads, q_len, head_dim)
        query_states = compute_query_states(model=self.model, **kwargs)

        previous_dtype = query_states.dtype

        # (bsz, num_heads, q_len, adapter_len)
        scores = torch.matmul(query_states, adapter_k.transpose(2, 3).to(previous_dtype)) / math.sqrt(
            self.model.head_dim
        )
        # Upcast attention to fp32
        # (bsz, num_heads, q_len, adapter_len)
        scores = self.adaption_gate * F.softmax(scores, dim=-1, dtype=torch.float32).to(previous_dtype)
        # (bsz, q_len, num_heads * head_dim)
        adapter_output = torch.matmul(scores, adapter_v).transpose(1, 2).reshape(bsz, q_len, -1)
        
        # (bsz, q_len, hidden_size)
        if o_proj_layer is not None:
            adapter_output = getattr(self.model, o_proj_layer)(adapter_output)

        # Add adaption prompt output to original output.
        output = output + adapter_output

        # Restore original dtype.
        output = output.to(previous_dtype)
        return output, None, past_key_value
