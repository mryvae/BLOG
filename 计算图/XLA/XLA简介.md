# XLA

## 1 什么是XLA?

XLA（加速线性代数）是一种针对特定领域的线性代数编译器，能够加快 TensorFlow 模型的运行速度，而且可能完全不需要更改源代码。

### 1.1 示例

```python
def model_fn(x, y, z):
  return tf.reduce_sum(x + y * z)
```

如果在不使用 XLA 的情况下运行，图会启动三个内核：分别对应于乘法、加法和减法运算。但是，XLA 可以优化该图，使其启动一次内核就能计算结果。它通过将加法、乘法和减法“融合”到一个 GPU 内核中来实现这一点。此外，这种融合操作不会将由 `y*z` 和 `x+y*z` 生成的中间值写出到内存中；而是直接将这些中间计算的结果“流式传输”给用户，同时将它们完全保留在 GPU 寄存器中。融合是 XLA 采用的最重要的一项优化措施。 内存带宽通常是硬件加速器上最稀缺的资源，因此消除内存操作是提高性能的最佳方法之一。

## 2 TensorFlow 模型启用 XLA

### 2.1 明确编译

通过 [`tf.function(jit_compile=True)`](https://tensorflow.google.cn/api_docs/python/tf/function?hl=zh-cn) 进行明确编译，借助 Explicit compilation API，可以精细地控制应选择哪些函数进行编译。

`jit_compile` API 具有必须编译语义：要么使用 XLA 编译整个函数，要么抛出 [`errors.InvalidArgumentError`](https://tensorflow.google.cn/api_docs/python/tf/errors/InvalidArgumentError?hl=zh-cn) 异常。目前，如果维度无法推断出来，也就是说，无法在不运行完整计算的情况下推断所有张量的维度，那么 XLA 无法编译包含此维度的函数。

### 2.2 自动聚类

若要在 TensorFlow 模型中开始使用 XLA 并且无需做出任何更改，最简单的方法是启用“自动聚类”，此功能会自动在可以使用 XLA 进行编译和执行的 TensorFlow 函数中查找聚类（连通的子图）。

## 3 XLA 架构

XLA 的输入语言称为“HLO IR”或仅为“HLO”（高级运算）。[运算语义](https://tensorflow.google.cn/xla/operation_semantics?hl=zh-cn)页面中介绍了 HLO 的语义。可以将 HLO 简单理解为[编译器 IR](https://en.wikipedia.org/wiki/Intermediate_representation)。

下图显示了 XLA 中的编译过程：

![img](https://tensorflow.google.cn/xla/images/how-does-xla-work.png?hl=zh-cn)

XLA 提供了多种与目标无关的优化和分析过程（例如 [CSE](https://en.wikipedia.org/wiki/Common_subexpression_elimination)）、与目标无关的运算融合，以及用于为计算分配运行时内存的缓冲区分析。

完成与目标无关的步骤之后，XLA 会将 HLO 计算发送到后端。后端可以执行进一步的 HLO 级优化，而此时将考虑目标特定的信息和需求。例如，XLA GPU 后端可以执行特别有利于 GPU 编程模型的运算融合，并确定如何将计算划分为计算流。在此阶段，后端还可能对某些运算或运算组合针对优化库调用执行模式匹配。

下一步是针对特定目标生成代码。XLA 所含的 CPU 和 GPU 后端使用 [LLVM](http://llvm.org/) 进行低级 IR、优化和代码生成。这些后端发出有效表示 XLA HLO 计算所需的 LLVM IR，然后调用 LLVM 以从此 LLVM IR 中发出原生代码。

## 4 reference

https://zhuanlan.zhihu.com/p/386425608

https://mp.weixin.qq.com/s/RO3FrPxhK2GEoDCGE9DXrw