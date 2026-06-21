from __future__ import annotations
import torch
from torch import device
class GraphModule(torch.nn.Module):
    def forward(self, output_11: "f16[s72, 12, 64]", s72: "Sym(s72)", l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_: "f16[768, 768]", l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_: "f16[768]", hidden_states_25: "f16[s72, 768]", l_self_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_: "f16[768]", l_self_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_: "f16[768]", l_self_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_: "f16[3072, 768]", l_self_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_: "f16[3072]", l_self_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_: "f16[768, 3072]", l_self_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_: "f16[768]", l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_: "f16[768]", l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_: "f16[768]", l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_qkv_proj_parameters_weight_: "f16[2304, 768]", l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_qkv_proj_parameters_bias_: "f16[2304]"):
        # File: /scratch/deepseek-prof/src/vllm/vllm/model_executor/layers/attention/attention.py:544 in forward, code: return output.view(-1, hidden_size)
        view: "f16[s72, 768]" = output_11.view(-1, 768);  output_11 = None

        # File: /scratch/deepseek-prof/src/vllm/vllm/model_executor/parameter.py:126 in __torch_function__, code: return super().__torch_function__(func, types, args, kwargs)
        linear: "f16[s72, 768]" = torch._C._nn.linear(view, l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_, l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_);  view = l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = None

        # File: /scratch/deepseek-prof/src/vllm/vllm/model_executor/models/opt.py:180 in forward, code: hidden_states = residual + hidden_states
        add: "f16[s72, 768]" = hidden_states_25 + linear;  hidden_states_25 = linear = None

        # File: /scratch/deepseek-prof/src/vllm/vllm/model_executor/models/opt.py:189 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        layer_norm: "f16[s72, 768]" = torch.nn.functional.layer_norm(add, (768,), l_self_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_, l_self_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_, 1e-05);  l_self_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_ = None

        # File: /scratch/deepseek-prof/src/vllm/vllm/model_executor/parameter.py:126 in __torch_function__, code: return super().__torch_function__(func, types, args, kwargs)
        linear_1: "f16[s72, 3072]" = torch._C._nn.linear(layer_norm, l_self_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_, l_self_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_);  layer_norm = l_self_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_ = None

        # File: /scratch/deepseek-prof/src/vllm/vllm/model_executor/models/opt.py:191 in forward, code: hidden_states = self.activation_fn(hidden_states)
        relu: "f16[s72, 3072]" = torch.nn.functional.relu(linear_1, inplace = False);  linear_1 = None

        # File: /scratch/deepseek-prof/src/vllm/vllm/model_executor/parameter.py:126 in __torch_function__, code: return super().__torch_function__(func, types, args, kwargs)
        linear_2: "f16[s72, 768]" = torch._C._nn.linear(relu, l_self_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_, l_self_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_);  relu = l_self_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_ = None

        # File: /scratch/deepseek-prof/src/vllm/vllm/model_executor/models/opt.py:193 in forward, code: hidden_states = residual + hidden_states
        add_1: "f16[s72, 768]" = add + linear_2;  add = linear_2 = None

        # File: /scratch/deepseek-prof/src/vllm/vllm/model_executor/models/opt.py:178 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        layer_norm_1: "f16[s72, 768]" = torch.nn.functional.layer_norm(add_1, (768,), l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_, l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_, 1e-05);  l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_ = None

        # File: /scratch/deepseek-prof/src/vllm/vllm/model_executor/parameter.py:126 in __torch_function__, code: return super().__torch_function__(func, types, args, kwargs)
        linear_3: "f16[s72, 2304]" = torch._C._nn.linear(layer_norm_1, l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_qkv_proj_parameters_weight_, l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_qkv_proj_parameters_bias_);  layer_norm_1 = l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_qkv_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_6_modules_self_attn_modules_qkv_proj_parameters_bias_ = None

        # File: /scratch/deepseek-prof/src/vllm/vllm/model_executor/models/opt.py:121 in forward, code: q, k, v = qkv.chunk(chunks=3, dim=-1)
        chunk = linear_3.chunk(chunks = 3, dim = -1);  linear_3 = None
        getitem: "f16[s72, 768]" = chunk[0]
        getitem_1: "f16[s72, 768]" = chunk[1]
        getitem_2: "f16[s72, 768]" = chunk[2];  chunk = None

        # File: /scratch/deepseek-prof/src/vllm/vllm/model_executor/layers/attention/attention.py:493 in forward, code: output = torch.empty(output_shape, dtype=output_dtype, device=query.device)
        size = torch.Size([s72, 768]);  s72 = None
        empty: "f16[s72, 768]" = torch.empty(size, dtype = torch.float16, device = device(type='cuda', index=0));  size = None

        # File: /scratch/deepseek-prof/src/vllm/vllm/model_executor/layers/attention/attention.py:498 in forward, code: query = query.view(-1, self.num_heads, self.head_size)
        view_1: "f16[s72, 12, 64]" = getitem.view(-1, 12, 64);  getitem = None

        # File: /scratch/deepseek-prof/src/vllm/vllm/model_executor/layers/attention/attention.py:499 in forward, code: output = output.view(-1, self.num_heads, self.head_size_v)
        view_2: "f16[s72, 12, 64]" = empty.view(-1, 12, 64);  empty = None

        # File: /scratch/deepseek-prof/src/vllm/vllm/model_executor/layers/attention/attention.py:501 in forward, code: key = key.view(-1, self.num_kv_heads, self.head_size)
        view_3: "f16[s72, 12, 64]" = getitem_1.view(-1, 12, 64);  getitem_1 = None

        # File: /scratch/deepseek-prof/src/vllm/vllm/model_executor/layers/attention/attention.py:503 in forward, code: value = value.view(-1, self.num_kv_heads, self.head_size_v)
        view_4: "f16[s72, 12, 64]" = getitem_2.view(-1, 12, 64);  getitem_2 = None
        return (view_3, view_4, view_1, view_2, add_1)
