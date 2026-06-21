
import os
os.environ['TORCHDYNAMO_VERBOSE'] = '0'
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCH_LOGS'] = '+dynamo,graph_breaks,recompiles,guards,aot_graphs,output_code,kernel_code'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/scratch/deepseek-prof/cache/torchinductor_lab'
os.environ['PYTORCH_KERNEL_CACHE_PATH'] = '/scratch/deepseek-prof/cache/torch/kernels'
os.environ['TRITON_CACHE_DIR'] = '/scratch/deepseek-prof/cache/triton_lab'
os.environ['TORCH_HOME'] = '/scratch/deepseek-prof/cache/torch'
os.environ['TORCH_LOGS_OUT'] = '/scratch/deepseek-prof/profiles/torch_dynamo_inductor/torch_logs.txt'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.replay_side_effects = True
torch._dynamo.config.side_effect_replay_policy = 'info'
torch._dynamo.config.specialize_int = False
torch._dynamo.config.specialize_float = False
torch._dynamo.config.assume_static_by_default = True
torch._dynamo.config.automatic_dynamic_shapes = True
torch._dynamo.config.capture_scalar_outputs = False
torch._dynamo.config.capture_dynamic_output_shape_ops = False
torch._dynamo.config.prefer_deferred_runtime_asserts_over_guards = False
torch._dynamo.config.do_not_emit_runtime_asserts = False
torch._dynamo.config.allow_rnn = False
torch._inductor.config.deterministic = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.selective_decompose = False



isolate_fails_code_str = None





if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
# torch version: 2.11.0+cu128
# torch cuda version: 12.8
# torch git version: 70d99e998b4955e0049d13a98d77ae1b14db1f45


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA A10 : 4 

torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.reset_table()

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()



    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1):
        convert_element_type = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        sub = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, arg0_1);  mul = arg0_1 = None
        add_1 = torch.ops.aten.add.Tensor(mul_1, arg1_1);  mul_1 = arg1_1 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(add_1, torch.float16);  add_1 = None
        permute = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        view = torch.ops.aten.view.default(convert_element_type_1, [1024, 1024]);  convert_element_type_1 = None
        mm = torch.ops.aten.mm.default(view, permute);  view = permute = None
        view_1 = torch.ops.aten.view.default(mm, [8, 128, 4096]);  mm = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(convert_element_type_4, convert_element_type_4)
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, convert_element_type_4);  mul_2 = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_3, 0.044715);  mul_3 = None
        add_2 = torch.ops.aten.add.Tensor(convert_element_type_4, mul_4);  mul_4 = None
        mul_5 = torch.ops.aten.mul.Tensor(add_2, 0.7978845608028654);  add_2 = None
        mul_6 = torch.ops.aten.mul.Tensor(convert_element_type_4, 0.5);  convert_element_type_4 = None
        tanh = torch.ops.aten.tanh.default(mul_5);  mul_5 = None
        add_3 = torch.ops.aten.add.Tensor(tanh, 1);  tanh = None
        mul_7 = torch.ops.aten.mul.Tensor(mul_6, add_3);  mul_6 = add_3 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(mul_7, torch.float16);  mul_7 = None
        permute_1 = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        view_2 = torch.ops.aten.view.default(convert_element_type_5, [1024, 4096]);  convert_element_type_5 = None
        mm_1 = torch.ops.aten.mm.default(view_2, permute_1);  view_2 = permute_1 = None
        view_3 = torch.ops.aten.view.default(mm_1, [8, 128, 1024]);  mm_1 = None
        add_4 = torch.ops.aten.add.Tensor(arg2_1, view_3);  arg2_1 = view_3 = None
        return (add_4,)

def load_args(reader):
    buf0 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf0, (1024,), dtype=torch.float16, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1, (1024,), dtype=torch.float16, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf2, (8, 128, 1024), dtype=torch.float16, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 8388608, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf3, (4096, 1024), dtype=torch.float16, is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 8388608, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf4, (1024, 4096), dtype=torch.float16, is_leaf=True)  # arg4_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)