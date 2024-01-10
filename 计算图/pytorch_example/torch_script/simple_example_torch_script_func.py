import torch 

def origin_func(x): 
    x = x**2
    x = x**3
    return x
x = torch.rand(1, 2, 3, 4)
jit_model = torch.jit.trace(origin_func, x)
print(jit_model.graph)

# 匹配用的子图定义，注意常量必须为[value=2]属性 
pattern = """ 
    graph(%x): 
        %const_2 = prim::Constant[value=2]() 
        %out = aten::pow(%x, %const_2) 
        return (%out) 
""" 
 
# 替换用的子图定义 
replacement = """ 
    graph(%x): 
        %out = aten::mul(%x, %x) 
        return (%out) 
""" 

 # 使用刚才定义的 pattern与replacement来编辑graph 
torch._C._jit_pass_custom_pattern_based_rewrite_graph(pattern, replacement, 
                                                      jit_model.graph) 
 
# 结果可视化，pow(x,2)被正确替换为mul(x,x)，pow(x,3)则保留原样不受影响。

print(jit_model.graph)