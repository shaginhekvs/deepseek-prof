


def forward(self, p_ln_weight, p_ln_bias, p_w1_weight, p_w2_weight, x):
    layer_norm = torch.ops.aten.layer_norm.default(x, [1024], p_ln_weight, p_ln_bias, 1e-05, False);  p_ln_weight = p_ln_bias = None
    linear = torch.ops.aten.linear.default(layer_norm, p_w1_weight);  layer_norm = p_w1_weight = None
    gelu = torch.ops.aten.gelu.default(linear, approximate = 'tanh');  linear = None
    linear_1 = torch.ops.aten.linear.default(gelu, p_w2_weight);  gelu = p_w2_weight = None
    add = torch.ops.aten.add.Tensor(x, linear_1);  x = linear_1 = None
    return (add,)
    