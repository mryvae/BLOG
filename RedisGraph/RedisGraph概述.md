### 1 RedisGraph简介

RedisGraph是高性能内存数据库Redis的图模块，它由Redis实验室开发，用于向Redis添加图形数据库功能。RedisGraph创新地将图数据表示为稀疏矩阵并利用GraphBLAS将图形操作转换为对矩阵的操作，同时还保留了Redis的单线程运行和完全基于内存的特点，这些特别之处为RedisGraph带来了独特的性能优势。

redis module是一种动态库，可以用与redis内核相似的运行速度和特性来扩展redis内核的功能；在redis中使用lua脚本只是组合Redis内核的现有功能，但是redis module则可以给redis内核添加新的功能。redis module可以将很多重复性的工作独立出来，交给特定的团队进行开发和维护，能够减少程序之间的耦合性；同时也能极大的提高开发效率，降低程序的维护开销。

RedisGraph is a graph database module for Redis.

### 2 Graph数据结构

RedisGraph使用稀疏邻接矩阵来表示图。RedisGraph选择将图表示为稀疏矩阵的主要原因之一是图遍历。

#### 2.1 csr_matrix

RedisGraph选择使用稀疏邻接矩阵`csr_matrix`来表示图。 `csr_matrix`，全称Compressed Sparse Row matrix，即按行压缩的稀疏矩阵存储方式，由三个一维数组`indptr`, `indices`, `data`组成。这种格式要求矩阵元**「按行顺序存储」**，**「每一行中的元素可以乱序存储」**。那么对于每一行就只需要用一个指针表示该行元素的起始位置即可。`indptr`存储每一行数据元素的起始位置，`indices`这是存储每行中数据的列号，与`data`中的元素一一对应。csr_matrix允许快速访问矩阵的行，但访问列的速度非常慢。

<img src="https://user-images.githubusercontent.com/83715643/197320952-6a74ec60-f32f-4700-bc4b-94abb40bb1ee.png" style="zoom:50%;" />

- csr_matrix的优点：

  高效的算术运算CSR + CSR，CSR × CSR等
  高效的行切片
  快速矩阵运算

- csr_matrix的缺点：

  列切片操作比较慢
  稀疏结构的转换比较慢

#### 2.2 Graph in RedisGraph

我们首先看一下RedisGraph中`Graph`数据结构的定义：

```c
struct Graph {
	DataBlock *nodes;                   // graph nodes stored in blocks
	DataBlock *edges;                   // graph edges stored in blocks
	RG_Matrix adjacency_matrix;         // adjacency matrix, holds all graph connections
	RG_Matrix *labels;                  // label matrices
	RG_Matrix node_labels;              // mapping of all node IDs to all labels possessed by each node
	RG_Matrix *relations;               // relation matrices
	RG_Matrix _zero_matrix;             // zero matrix
	pthread_rwlock_t _rwlock;           // read-write lock scoped to this specific graph
	bool _writelocked;                  // true if the read-write lock was acquired by a writer
	SyncMatrixFunc SynchronizeMatrix;   // function pointer to matrix synchronization routine
	GraphStatistics stats;              // graph related statistics
};
```

- `Graph`数据结构维护三个矩阵：

  矩阵均为NxN的方阵，N为顶点数。我们考虑具有visits和friend两种关系类型的图。

  - adjacency_matrix

    邻接矩阵会标记图中的所有关系连接，关系类型不可知。

  - labels

    为了适应类型化节点，每个标签分配一个额外的矩阵，并且标签矩阵与沿主对角线的矩阵对称。

    假设节点 N 被标记为 Person，那么 Person对应的label 矩阵 P 会将位置 P[N,N] 设置为 1。

  - relations

    每个类型的关系都有自己的专用矩阵。

    对于visits和friend两种关系，会存在对应的visits和friend关系矩阵。

    如果节点A和B存在friend关系，则friend[A,B]=1

- 这种设计让RedisGraph可以轻松修改其图形，包括：
  - 添加新节点只是扩展矩阵，添加额外的行和列
  - 通过在相关矩阵中设置相关条目来添加新关系
  - 删除关系会清除相关条目
  - 通过删除矩阵行/列来删除节点。

### 3 execution in RedisGraph

RedisGraph使用Cypher查询语言，并为其构建了解析器。与一般的关系数据库类似，RedisGraph也需要进行词法分析、语法分析、语义分析等流程，从而生成执行计划。RedisGraph会将查询操作转换为相应的矩阵操作，获取查询结果。

<img src="https://user-images.githubusercontent.com/83715643/192198026-6070786b-30de-4178-ba51-ce1021cb7063.png" alt="image" style="zoom:50%;" />

<img src="https://user-images.githubusercontent.com/83715643/192198286-1f576588-7224-4fbb-a5a8-b3020f27f39f.png" alt="img" style="zoom: 50%;" />


#### 3.1 遍历

图遍历是通过矩阵乘法完成的。例如，如果我们想为图中的每个节点找到朋友的朋友，这个遍历可以表示为 FOF = F^2。F代表friend关系矩阵，矩阵FOF中保存了遍历结果。FOF 的行代表源节点，其列代表两跳外的朋友：如果 FOF[i,j] = 1，则 j 是 i 的朋友的朋友。

#### 3.2 搜索模式

当一个搜索模式`(N0)-[A]->(N1)-[B]->(N2)<-[A]-(N3)`被用作查询的一部分时，我们将其转换为一组矩阵乘法。对于给定的示例，一种可能的表达式是：`A * B * Transpose(A)`.

请注意，矩阵乘法是一种关联和分配运算。这使我们可以自由选择首先要相乘的项（首选会产生高度稀疏的中间矩阵的项）。它还可以在计算表达式时实现并发。

### 4 reference

- RedisGraph，https://oss.redislabs.com/redisgraph/
- RedisGraph，https://redis.io/docs/stack/graph/design/
- RedisGraph，https://github.com/RedisGraph/RedisGraph
- RedisConf18 - Lower Latency Graph Queries in Cypher with Redis Graph，https://www.slideshare.net/RedisLabs/redisconf18-lower-latency-graph-queries-in-cypher-with-redis-graph
- RedisGraph，https://zhuanlan.zhihu.com/p/100996492
- csc_matrix，https://zhuanlan.zhihu.com/p/456904535
