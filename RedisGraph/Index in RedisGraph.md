 ### Index in RedisGraph

#### 概述

索引是图数据库设计和开发的重要方面。索引可以大大提升数据库的查询效率。

举个例子，当RedisGraph处理以下请求(query)：

```cypher
GRAPH.QUERY DEMO_GRAPH "MATCH (n:作品)-[]-(m:角色) where m.name='郭靖' return n.name"
```

如果没有建立相应的索引，那RedisGraph将不得不对label是角色的节点进行遍历，查询效率低下；如果存在相应的索引，RedisGraph可以根据name='郭靖'直接找到目标节点，避免低效的遍历操作。

 #### Overview of index in RedisGraph

RedisGraph中并没有索引的代码实现，其使用RediSearch搜索引擎来构造索引。

RediSearch是一个高性能的全文搜索引擎，可作为一个Redis Module 运行在Redis上。RediSearch的核心数据结构是Inverted index，也就是hashtable。也就是说，RedisGraph中的索引为哈希索引。

RedisGraph中的索引分为两种：

- exact match index

  RedisGraph仅支持对node/edge建立单属性索引。

  比如，我们可以在label是角色的节点上为name属性建立单属性索引：

  ```cypher
  GRAPH.QUERY DEMO_GRAPH "CREATE INDEX ON :Person(age)"
  ```

- full-text index

  RedisGraph利用RediSearch的索引功能，通过procedure call提供全文索引。

  比如，我们可以在label为“角色”的所有节点的name属性上构造全文索引：

  ```cypher
  GRAPH.QUERY DEMO_GRAPH "CALL db.idx.fulltext.createNodeIndex('Person', 'name')"
  ```

#### Implementation of index in RedisGraph

首先，我们来看一下`Index`的实现。`Index`存在两种类型，即为exact-match 、 fulltext，其支持对node/edge建立单属性索引。RedisGraph使用RediSearch搜索引擎来构造索引，`Index`封装了RedisGraph中`RSIndex`。

```c
typedef struct {
	char *label;                  // indexed label
	int label_id;                 // indexed label ID
	IndexField *fields;           // indexed fields
	char *language;               // language
	char **stopwords;             // stopwords
	GraphEntityType entity_type;  // entity type (node/edge) indexed
	IndexType type;               // index type exact-match / fulltext
	RSIndex *idx;                 // rediSearch index
} Index;
```


RedisGraph使用`Schema`来组织索引。对于每一种entity (node/edge)，RedisGraph均会维护一个`Schema`。`Schema`中存储着与该entity (node/edge)相关的所有索引。

```c
typedef struct {
	int id;               // schema id
	char *name;           // schema name
	SchemaType type;      // schema type (node/edge)
	Index *index;         // exact match index
	Index *fulltextIdx;   // full-text index
} Schema;
```

