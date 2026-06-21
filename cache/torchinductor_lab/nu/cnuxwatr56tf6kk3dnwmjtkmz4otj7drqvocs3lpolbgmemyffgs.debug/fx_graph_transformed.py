class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[1024]", arg1_1: "f16[1024]", arg2_1: "f16[8, 128, 1024]", arg3_1: "f16[4096, 1024]", arg4_1: "f16[1024, 4096]"):
        # File: /scratch/deepseek-prof/scripts/torch_dynamo_inductor_lab.py:24 in forward, code: x = self.ln(x)
        convert_element_type: "f32[8, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 128, 1]" = var_mean[0]
        getitem_1: "f32[8, 128, 1]" = var_mean[1];  var_mean = None
        sub: "f32[8, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[8, 128, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[8, 128, 1024]" = torch.ops.aten.mul.Tensor(mul, arg0_1);  mul = arg0_1 = None
        add_1: "f32[8, 128, 1024]" = torch.ops.aten.add.Tensor(mul_1, arg1_1);  mul_1 = arg1_1 = None
        convert_element_type_1: "f16[8, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_1, torch.float16);  add_1 = None

        # File: /scratch/deepseek-prof/scripts/torch_dynamo_inductor_lab.py:25 in forward, code: x = torch.nn.functional.gelu(self.w1(x), approximate="tanh")
        view: "f16[1024, 1024]" = torch.ops.aten.reshape.default(convert_element_type_1, [1024, 1024]);  convert_element_type_1 = None
        permute: "f16[1024, 4096]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        mm: "f16[1024, 4096]" = torch.ops.aten.mm.default(view, permute);  view = permute = None
        view_1: "f16[8, 128, 4096]" = torch.ops.aten.reshape.default(mm, [8, 128, 4096]);  mm = None
        convert_element_type_4: "f32[8, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        mul_6: "f32[8, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 0.5)
        mul_2: "f32[8, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_4, convert_element_type_4)
        mul_3: "f32[8, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_2, convert_element_type_4);  mul_2 = None
        mul_4: "f32[8, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_3, 0.044715);  mul_3 = None
        add_2: "f32[8, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_4, mul_4);  convert_element_type_4 = mul_4 = None
        mul_5: "f32[8, 128, 4096]" = torch.ops.aten.mul.Tensor(add_2, 0.7978845608028654);  add_2 = None
        tanh: "f32[8, 128, 4096]" = torch.ops.aten.tanh.default(mul_5);  mul_5 = None
        add_3: "f32[8, 128, 4096]" = torch.ops.aten.add.Tensor(tanh, 1);  tanh = None
        mul_7: "f32[8, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_6, add_3);  mul_6 = add_3 = None
        convert_element_type_5: "f16[8, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_7, torch.float16);  mul_7 = None

        # File: /scratch/deepseek-prof/scripts/torch_dynamo_inductor_lab.py:26 in forward, code: return residual + self.w2(x)
        view_2: "f16[1024, 4096]" = torch.ops.aten.reshape.default(convert_element_type_5, [1024, 4096]);  convert_element_type_5 = None
        permute_1: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        mm_1: "f16[1024, 1024]" = torch.ops.aten.mm.default(view_2, permute_1);  view_2 = permute_1 = None
        view_3: "f16[8, 128, 1024]" = torch.ops.aten.reshape.default(mm_1, [8, 128, 1024]);  mm_1 = None
        add_4: "f16[8, 128, 1024]" = torch.ops.aten.add.Tensor(arg2_1, view_3);  arg2_1 = view_3 = None
        return (add_4,)
