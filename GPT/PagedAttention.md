# PagedAttention

## 1 Overview

PagedAttention灵感来自于操作系统中虚拟内存和分页的经典思想，它可以允许在非连续空间立存储连续的KV张量。具体来说，PagedAttention把每个序列的KV缓存进行了分块，每个块包含固定长度的token，而在计算attention时可以高效地找到并获取那些块。

<img src="https://pic2.zhimg.com/v2-6035b0440dd9f0eb37bc9c221b977799_b.webp" alt="动图" style="zoom:50%;" />

## 2 Block management

相比于RaggedAttention，PagedAttention其实就是维护了一个逻辑block到物理block的映射。接下来，我们结合代码，看一下是vLLM如何实现PagedAttention中Block管理的。



## 3 PagedAttention kernal

### 3.1 attention 伪代码

### 3.2 paged_attention_v1_kernel

### 3.3 paged_attention_v2_kernel

