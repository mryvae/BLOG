 ### ConcurrencyControl in RedisGraph

RedisGraph是Redis的图数据库模块。Redis是一个高性能的内存数据库，其使用单线程模型，即Redis的网络IO和键值对读写是由一个线程来完成的。RedisGraph保留了Redis完全基于内存的特点。然而，RedisGraph并没有保留Redis的单线程模型，其内部创建了线程池，用于加快读写请求的处理，这就需要相应的并发控制机制。

#### threadpool in RedisGraph

RedisGraph内创建有两个线程池，分别是读线程池(`_readers_thpool`)和写线程池(`_writers_thpool`)，分别用于处理读写请求。当接收到query时，redisgraph会为其分配读写线程，若线程池满，则放入等待队列中。

- `_readers_thpool`
  - 在默认配置下，读线程池中线程数量为3，其等待队列大小为7。
  - 读线程池的大小及其队列大小均可根据使用需求重新配置。
  
- `_writers_thpool`
  - 在默认配置下，写线程池中线程数量为1，其等待队列大小为7。
  - 写线程池中线程数量不可重新配置；
  - 写线程池中等待队列长度可根据使用需求重新配置。

#### Lock in RedisGraph

Redisgraph使用加锁来实现并发控制。在并发控制过程中，Redisgraph中的存在锁的数据结构主要有两个：`Graph`、`GraphContext`。

- Lock in `Graph`

  `Graph`为RedisGraph中的核心数据结构之一，用于存储图模型。

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
  
  在并发控制过程中，Redisgraph的封锁粒度为`Graph`。
  
  - ReadLock
  
    对于read-only的请求，Redisgraph会在执行ExecutionPlan期间封锁整个Graph。当ExecutionPlan执行完毕，释放ReadLock。
  
  - WriteLock
  
    对于write请求，Redisgraph会在commit更新索引(index)或entity(node/edge)的操作的时候封锁整个Graph。当commit操作完成，释放WriteLock。

- Lock in `GraphContext`

  顾名思义，`GraphContext`为`Graph`上下文。其中存储与`Graph`相关的索引、日志等内容。

  ```c
  typedef struct {
  	Graph *g;                               // container for all matrices and entity properties
  	int ref_count;                          // number of active references
  	rax *attributes;                        // from strings to attribute IDs
  	pthread_rwlock_t _attribute_rwlock;     // read-write lock to protect access to the attribute maps
  	char *graph_name;                       // string associated with graph
  	char **string_mapping;                  // from attribute IDs to strings
  	Schema **node_schemas;                  // array of schemas for each node label
  	Schema **relation_schemas;              // array of schemas for each relation type
  	unsigned short index_count;             // number of indicies
  	SlowLog *slowlog;                       // slowlog associated with graph
  	GraphEncodeContext *encoding_context;   // encode context of the graph
  	GraphDecodeContext *decoding_context;   // decode context of the graph
  	Cache *cache;                           // global cache of execution plans
  	XXH32_hash_t version;                   // graph version
  } GraphContext;
  ```

  `GraphContext.attributes`中存储着attribute名称(type:string)到attribute编号(type:unsigned short)的映射。

  - ReadLock

    当线程读取`GraphContext.attributes`中内容时，需要获取ReadLock。

  - WriteLock

    当线程修改`GraphContext.attributes`中内容时，需要获取WriteLock。