## Multiplication of sparse Matrices

### 1 Problem Definition

稀疏矩阵乘法问题是计算$C＝AB$，其中分别具有$m×k$和$k×n$维度的输入矩阵$A$和$B$都是稀疏的。

我们认为矩阵$A$是稀疏的，当且仅当$nnz(A)=O(max(m,n))$。其中，$nnz(A)$为矩阵$A$中非零元素的个数，$m$为矩阵$A$的行数，$n$为矩阵$A$的列数。

输入矩阵以某种节省空间的格式表示，输出矩阵C也应采用与A和B相同的格式。在大多数稀疏矩阵包（包括Matlab）中存储稀疏矩阵的数据结构是csc_matrix。

### 2 Sparse matrix format

本文关注两种稀疏矩阵格式：csr_matrix、csc_matrix。

#### 2.1 csr_matrix

`csr_matrix`，全称Compressed Sparse Row matrix，即按行压缩的稀疏矩阵存储方式，由三个一维数组`indptr`, `indices`, `data`组成。这种格式要求矩阵元**「按行顺序存储」**，**「每一行中的元素可以乱序存储」**。那么对于每一行就只需要用一个指针表示该行元素的起始位置即可。`indptr`存储每一行数据元素的起始位置，`indices`这是存储每行中数据的列号，与`data`中的元素一一对应。csr_matrix允许快速访问矩阵的行，但访问列的速度非常慢。

<img src="https://user-images.githubusercontent.com/83715643/197320952-6a74ec60-f32f-4700-bc4b-94abb40bb1ee.png" style="zoom:50%;" />

#### 2.2 csc_matrix

`csr_matrix`，全称Compressed Sparse Column matrix，即按列压缩的稀疏矩阵存储方式，由三个一维数组`indptr`, `indices`, `data`组成。这种格式要求矩阵元**「按列顺序存储」**，**「每一列中的元素可以乱序存储」**。那么对于每一列就只需要用一个指针表示该列元素的起始位置即可。`indptr`存储每一列数据元素的起始位置，`indices`这是存储每行中数据的行号，与`data`中的元素一一对应。csc_matrix允许快速访问矩阵的列，但访问行的速度非常慢。

<img src="https://user-images.githubusercontent.com/83715643/191659786-f067a310-ff2d-4bd8-b2f4-2d2f3a400373.png" alt="image" style="zoom:50%;" />

csc_matrix和csr_matrix的区别在于，csc_matrix选择按列压缩的稀疏矩阵存储方式，而csr_matrix是按行存储。

### 3 Multiplication Algorithms

#### 3.1 Algorithm1

我们首先介绍Matlab中的稀疏矩阵乘算法。Matlab中稀疏矩阵格式是csc_matrix。

使用MATLAB语言表示算法，矩阵$C$第$j$列计算为：

```matlab
for k = find B(:,j)
	C(:,j) = C(:,j) + B(k,j)A(:,k)
end
```

其中，$A$和$B$是具有$m×k$和$k×n$维度的csc_matrix。`find B(:,j)`表示查询矩阵$B$第$j$列中下一个非零元素的行索引。

在计算矩阵$C$第$j$列的过程中，需要一个辅助数据结构SPA(the sparse accumulator)。SPA由三个组件组成：一个包含C的活动列的实值的密集向量，另一个包含“占用”标志的密集布尔向量，以及一个包含当前活动列的非零元素索引的稀疏列表。SPA本身使用空间$O(m)$，因此其初始化需要$O(m)$时间。

```c
struct SPA{
    T value[m];
    bool flag[m];
    list nonzero_element;
}
```

矩阵$C$可以进行按列分块，每一个块分配一个线程，进行并行计算。

对于大量线程，Algorithm1无法很好地扩展。每一个线程均需要维护一个SPA数据结构，SPA空间复杂度为$O(m)$，初始化需要$O(m)$时间。一方面，大部分时间都用于SPA初始化操作；另一方面，SPA数据结构占用内存过大，因为需要为每一个线程分配一个SPA。

#### 3.2 Algorithm2

Algorithm2要求输入矩阵使用csr_matrix、csc_matrix两种稀疏矩阵格式存储，并且每一行/列中的元素必须顺序存储。

使用MATLAB语言表示算法，矩阵$C$计算为：

```matlab
for i = 1:k
	C(:,:) = C(:,:) + A(:,i)B(i,:)
end
```

其中，$A$和$B$是具有$m×k$和$k×n$维度的稀疏矩阵，均使用csr_matrix、csc_matrix两种稀疏矩阵格式存储。

在Algorithm2中，$A$的第$i$列和$B$的第$i$行相乘生成秩1矩阵。计算完所有的秩1矩阵（共$k$个）后，需要将$k$个秩1矩阵merge成一个结果矩阵$C$，$C$使用csr_matrix、csc_matrix两种稀疏矩阵格式存储。注意，秩1矩阵中的元素是按行顺序存储的。

- merge algorithm
  - 使用堆(heap)进行merge操作。
  - heap中的element的value为矩阵的元素，key为矩阵的元素相应的行列索引$(i,j)$。
  - heap的大小为$k$。
  - 重复地从heap中提取min_element，并在min_element所属的秩1矩阵中选择下一个非零元素min_element_next，将min_element_next插入heap。
  - 如果heap中有多个元素具有相同的key，那么它们的value会在运行中求和。

对于$k$个秩1矩阵的计算，可以分配多个线程进行并行计算。

#### 3.3 Algorithm3

Algorithm3要求输入矩阵使用csc_matrix稀疏矩阵格式存储，并且每一列中的元素必须顺序存储。

使用MATLAB语言表示算法，矩阵$C$第$j$列计算为：

```matlab
for k = find B(:,j)
	C(:,j) = C(:,j) + B(k,j)A(:,k)
end
```

Algorithm3与Algorithm1类似，两者不同之处在于，Algorithm1在计算矩阵$C$第$j$列的过程中，需要辅助数据结构SPA；Algorithm3在计算矩阵$C$第$j$列的过程中，需要辅助数据结构heap，heap的作用是将多个`A(:,k)`进行merge成`C(:,j)`。

- merge algorithm
  - 使用堆(heap)进行merge操作。
  - heap中的element的value为`A(:,k)`中的元素，key为矩阵的元素相应的行索引$i$，即`A(i,k)=value`。
  - heap的大小为$nnz(A(:,k))$。
  - 重复地从heap中提取min_element，并在min_element所属的`A(:,k)`中选择下一个非零元素min_element_next，将min_element_next插入heap。
  - 如果heap中有多个元素具有相同的key，那么它们的value会在运行中求和。

矩阵$C$可以进行按列分块，每一个块分配一个线程，进行并行计算。

