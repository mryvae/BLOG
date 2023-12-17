# Transformer

本文章仅仅是记录自己学习transformer模型的笔记。如果是小白，请移步B站观看台大李宏毅老师的经典讲解！

【强烈推荐！台大李宏毅自注意力机制和Transformer详解！】https://www.bilibili.com/video/BV1v3411r78R?p=2&vd_source=b5950193baa07837c4ead60578fb4c2b

## 1 Self-attention

### 1.1 overview

- 输入一个vector sequence $a^{1}, a^{2}, a^{3}, a^{4}$
- 输出一个vector sequence $b^{1}, b^{2}, b^{3}, b^{4}$
- 考虑vector sequence的相关性

<img src="https://github.com/mryvae/picture_bed/assets/83715643/ab69d9da-b991-4c08-a395-c69eeff2f3e9" style="zoom:33%;" />

### 1.2 考虑$b^{1}$的生成

- $\alpha '$为输入向量间的相关性
- $q=W^{q}a, v=W^{v}a, k=W^{k}a$
- $b^{i}=\sum_{i}^{} \alpha '_{1,i}v^{i}$
- 模型参数为三个矩阵$W^{q}, W^{k}, W^{v}$

<img src="https://github.com/mryvae/picture_bed/assets/83715643/fb24872a-ae20-4a82-86b4-48321865c810" style="zoom:33%;" />

### 1.3 写成矩阵形式

- $I=[a^{1}, a^{2}, a^{3}, a^{4}]$, $Q=[q^{1}, q^{2}, q^{3}, q^{4}]$, $K=[k^{1}, k^{2}, k^{3}, k^{4}]$, $V=[v^{1}, v^{2}, v^{3}, v^{4}]$, $O=[b^{1}, b^{2}, b^{3}, b^{4}]$
- $Q=W^{q}I$, $K=W^{k}I$, $V=W^{v}I$
- $A'=softmax(A)$, $A=K^{\mathrm{T}}Q$
- $O=VA'$

<img src="https://github.com/mryvae/picture_bed/assets/83715643/3594564c-df0a-4f6f-8de7-eeb1d3fbfb63" style="zoom:33%;" />

<img src="https://github.com/mryvae/picture_bed/assets/83715643/cc5b736d-f7e6-462b-bf7f-eda6a2bd6adb" style="zoom:33%;" />

<img src="https://github.com/mryvae/picture_bed/assets/83715643/02026754-e7f8-4551-830f-af97e6588e18" style="zoom:33%;" />

### 1.4 Multi-head

- 以2 head为例，仅仅是多了$W^{q,1}$, $W^{q,2}$, $W^{k,1}$, $W^{k,2}$,$W^{v,1}$, $W^{v,2}$, $W^{O}$这7个参数矩阵

<img src="https://github.com/mryvae/picture_bed/assets/83715643/14e208fb-35ac-40c2-b766-e560003a8139" style="zoom:33%;" />

<img src="https://github.com/mryvae/picture_bed/assets/83715643/b28c8cf0-3ae1-4717-b071-a34a050dbc88" style="zoom:33%;" />

## 2 Transformer

## 2.1 模型结构

- encoder 将一串输入嵌入
- decoder 负责解析输出结果

<img src="https://github.com/mryvae/picture_bed/assets/83715643/e60f9b93-ec41-4ab7-9e9e-e4ed727d8a6d" style="zoom:33%;" />

- 这张图是transformer的模型图
- 左半部分就是encoder，右半部分是decoder

<img src="https://github.com/mryvae/picture_bed/assets/83715643/b0f3a35e-cc94-4556-9176-57da62d1bf65" style="zoom: 50%;" />

## 2.2 encoder

- $x_{i}$经过N个Self-attention网络输出$b_i$
- 中间有全连接层，还是norm层

## 2.3 decoder

### 2.3.1 overview

- 上一个输出的字符作为模型的输入，从而产生下一个输出的字符

<img src="https://github.com/mryvae/picture_bed/assets/83715643/bd3f12ac-d737-4266-96f4-9a9aebc80de2" style="zoom:33%;" />

### 2.3.2 Masked Self-attention

- 与Self-attention相比，Masked Self-attention产生输出时会屏蔽一部分信息
- 比如，$b^{2}$只和$a^{1}, a^{2}$相关

<img src="https://github.com/mryvae/picture_bed/assets/83715643/7b3c1f53-6add-4bbb-a9be-49423c2844ee" style="zoom:33%;" />

### 2.3.3 Cross attention

- encoder 产生的输出与 经过Self-attention 的输入进行attention

<img src="https://github.com/mryvae/picture_bed/assets/83715643/cba37c11-67a0-4af4-8f43-2107610d5b36" style="zoom:33%;" />
