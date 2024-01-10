import torch 

@torch.jit.script
def origin_func(x): 
    x = x**2  # 乘法
    x = x**3  # 三次方
    return x 

print(origin_func.code)
print(origin_func.graph)
#%2 : int = prim::Constant[value=2]()
#%5 : int = prim::Constant[value=3]()
#%x.3 : Tensor = aten::pow(%x.1, %2)
#%x.5 : Tensor = aten::pow(%x.3, %5)
# 我们 希望 使用 乘法 来 替换 平方。

### 步骤1 注册子图替换 匹配模型 ###
# 匹配用的子图定义，注意常量必须为[value=2]属性
src_pattern = """
    graph(%x):
        %const_2 = prim::Constant[value=2]()
        %out = aten::pow(%x, %const_2)
        return (%out)
"""

# 替换用的子图定义 替换后目标图的模式
replacement = """
    graph(%x):
        %out = aten::mul(%x, %x)
        return (%out)
"""
# 使用刚才定义的 pattern与replacement来编辑graph 
torch._C._jit_pass_custom_pattern_based_rewrite_graph(src_pattern, replacement,
                                                    origin_func.graph)

# 结果可视化，pow(x,2)被正确替换为mul(x,x)，pow(x,3)则保留原样不受影响。
print(origin_func.graph) 
#%17 : Tensor = aten::mul(%x.1, %x.1)
#%5 : int = prim::Constant[value=3]()
#%x.5 : Tensor = aten::pow(%17, %5)

# 替换 pow(x,3)
src_pattern = """
    graph(%x):
        %const_3 = prim::Constant[value=3]()
        %out = aten::pow(%x, %const_3)
        return (%out)
"""

# 替换用的子图定义 替换后目标图的模式
replacement = """
    graph(%x):
        %tp = aten::mul(%x, %x)
        %out = aten::mul(%tp, %x)
        return (%out)
"""
torch._C._jit_pass_custom_pattern_based_rewrite_graph(src_pattern, replacement,
                                                    origin_func.graph)
print(origin_func.graph)
#%17 : Tensor = aten::mul(%x.1, %x.1)
#%18 : Tensor = aten::mul(%17, %17)
#%19 : Tensor = aten::mul(%18, %17)