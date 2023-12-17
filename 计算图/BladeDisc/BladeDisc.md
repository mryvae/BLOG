# BladeDISC 简介

## 1 BladeDISC是什么?

BladeDISC是阿里最新开源的基于MLIR的动态shape深度学习编译器。

- 专注于上层模型开发，降低手工优化性能的人力开发成本
- 把硬件的算力在真实的业务场景中发挥出来

### 1.1 主要特性

- 多款前端框架支持：TensorFlow，PyTorch
- 多后端硬件支持：CUDA，ROCM，x86
- 完备支持动态shape语义编译
- 支持推理及训练
- 轻量化API，对用户通用透明
- 支持插件模式嵌入宿主框架运行，以及独立部署模式

## 2 BladeDISC 架构

### 2.1 IR

MLIR 的核心是一套灵活的多层IR基础设施和编译器实用工具库。BladeDISC基于MLIR的主要原因包括其比较丰富的基础设施支持，方便扩展的模块化设计架构以及MLIR较强的胶水能力。

在动态shape场景下，IR本身需要有足够的能力表达shape计算和动态shape信息的传递。BladeDISC从项目建立开始一直与MHLO社区保持紧密的合作，在XLA的HLO IR基础上，扩展了一套具有完备动态shape表达能力的IR。

### 2.2 动态shape

为完备支持动态shape，编译结果需要能够在运行时做实时的shape推导计算，不仅要为数据计算，同时也需要为shape计算做代码生成。计算后的shape信息用于做内存/显存管理，以及kernel调度时的参数选择等等。

对于计算密集型部分，不同的shape要求更加精细的schedule实现来获得更好的性能，pass pipeline在设计上的主要考虑是需要支持在运行时根据不同的具体shape选择合适的算子库实现，以及处理动态shape语义下的layout问题。

而访存密集型部分的自动算子融合作为深度学习编译器主要的性能收益来源之一，同样面临shape未知情况下在性能上的挑战。

BladeDISC选择将部分的优化决策从编译时下沉到运行时。即在编译期根据一定的规则生成多个版本的kernel实现，在运行时根据实际shape自动选择最优的实现。

BladeDISC在IR及pass pipeline的设计过程中着重设计了shape constraint在IR中的抽象和在pass pipeline中的使用，在优化整体性能方面起到了比较明显的作用，保证能够足够接近甚至超过静态shape编译器的性能结果。

## 3 BladeDISC 使用

### 3.1 For TensorFlow Users

Only two lines of code are needed on native TensorFlow program as the following:

```python
import numpy as np
import tensorflow as tf

## enable BladeDISC on TensorFlow program
import blade_disc_tf as disc
disc.enable()

## construct TensorFlow Graph and run it
g = tf.Graph()
with g.as_default():
    ...
    with tf.session as sess:
        sess.run(...)
```

### 3.2 For PyTorch Users

PyTorch users only need the following few lines of code to enable BladeDISC:

```python
import torch_blade
# construct PyTorch Module
class MyModule(nn.Module):
    ...

module = MyModule().eval()

with torch.no_grad():
    # blade_module is the optimized module by BladeDISC
    blade_module = torch_blade.optimize(module, allow_tracing=True, model_inputs=(x, y))

# run the optimized module
blade_module(x, y)
```

`torch_blade.optimize` accepts an `nn.Module` object and outputs the optimized module.

## 4 reference

https://github.com/alibaba/BladeDISC

https://zhuanlan.zhihu.com/p/462641670