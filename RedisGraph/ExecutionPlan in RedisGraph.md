 ### ExecutionPlan in RedisGraph

```shell
127.0.0.1:6079> GRAPH.EXPLAIN scenario2 "MATCH (r:router)-[]->(t1)-[]->(t2)-[]->(t3)-[]->(t4) WHERE r.id >= 0 AND r.id < 4 RETURN t1,t2,t3,t4"
1) "Results"
2) "    Project"
3) "        Conditional Traverse | (t3)->(t4)"
4) "            Conditional Traverse | (t2)->(t3)"
5) "                Conditional Traverse | (t1)->(t2)"
6) "                    Conditional Traverse | (r:router)->(t1)"
7) "                        Node By Index Scan | (r:router)"
```

执行计划执行：

```c
ResultSet *ExecutionPlan_Execute(ExecutionPlan *plan) {
	ASSERT(plan->prepared)
	/* Set an exception-handling breakpoint to capture run-time errors.
	 * encountered_error will be set to 0 when setjmp is invoked, and will be nonzero if
	 * a downstream exception returns us to this breakpoint. */
	int encountered_error = SET_EXCEPTION_HANDLER();

	// Encountered a run-time error - return immediately.
	if(encountered_error) return QueryCtx_GetResultSet();

	ExecutionPlan_Init(plan);

	Record r = NULL;
	// Execute the root operation and free the processed Record until the data stream is depleted.
	while((r = OpBase_Consume(plan->root)) != NULL) ExecutionPlan_ReturnRecord(r->owner, r);

	return QueryCtx_GetResultSet();
}
```

执行计划的根节点为ResultsOp

```c
OpBase *NewResultsOp(const ExecutionPlan *plan) {
	Results *op = rm_malloc(sizeof(Results));

	// Set our Op operations
	OpBase_Init((OpBase *)op, OPType_RESULTS, "Results", ResultsInit, ResultsConsume,
				NULL, NULL, ResultsClone, NULL, false, plan);

	return (OpBase *)op;
}
```

创建根节点ResultsOp

```c
static ExecutionPlan *_ExecutionPlan_UnionPlans(AST *ast) {
	OpBase *results_op = NewResultsOp(plan);
}
```

ResultsConsume每次调用一次ProjectConsume，获取一个record

```c
static Record ResultsConsume(OpBase *opBase) {
	Record r = NULL;
	Results *op = (Results *)opBase;

	// enforce result-set size limit
	if(op->result_set_size_limit == 0) return NULL;
	op->result_set_size_limit--;

	OpBase *child = op->op.children[0];
	r = OpBase_Consume(child);
	if(!r) return NULL;

	// append to final result set
	ResultSet_AddRecord(op->result_set, r);
	return r;
}
```

ProjectConsume每次调用一次CondTraverseConsume，获取一个record。

```c
static Record ProjectConsume(OpBase *opBase) {
	OpProject *op = (OpProject *)opBase;

	if(op->op.childCount) {
		OpBase *child = op->op.children[0];
		op->r = OpBase_Consume(child);
		if(!op->r) return NULL;
	} else {
		// QUERY: RETURN 1+2
		// Return a single record followed by NULL on the second call.
		if(op->singleResponse) return NULL;
		op->singleResponse = true;
		op->r = OpBase_CreateRecord(opBase);
	}

	op->projection = OpBase_CreateRecord(opBase);

	for(uint i = 0; i < op->exp_count; i++) {
		AR_ExpNode *exp = op->exps[i];
		SIValue v = AR_EXP_Evaluate(exp, op->r);
		int rec_idx = op->record_offsets[i];
		/* Persisting a value is only necessary here if 'v' refers to a scalar held in Record 'r'.
		 * Graph entities don't need to be persisted here as Record_Add will copy them internally.
		 * The RETURN projection here requires persistence:
		 * MATCH (a) WITH toUpper(a.name) AS e RETURN e
		 * TODO This is a rare case; the logic of when to persist can be improved.  */
		if(!(v.type & SI_GRAPHENTITY)) SIValue_Persist(&v);
		Record_Add(op->projection, rec_idx, v);
		/* If the value was a graph entity with its own allocation, as with a query like:
		 * MATCH p = (src) RETURN nodes(p)[0]
		 * Ensure that the allocation is freed here. */
		if((v.type & SI_GRAPHENTITY)) SIValue_Free(v);
	}

	OpBase_DeleteRecord(op->r);
	op->r = NULL;

	// Emit the projected Record once.
	Record projection = op->projection;
	op->projection = NULL;
	return projection;
}
```

CondTraverseConsume在第一次被调用时，就会把所有的数据查询出来，放到一个RG_MatrixTupleIter中。之后，每一次调用CondTraverseConsume，RG_MatrixTupleIter向上层返回一个record。

```c
static Record CondTraverseConsume(OpBase *opBase) {
	OpCondTraverse *op = (OpCondTraverse *)opBase;
	OpBase *child = op->op.children[0];

	/* If we're required to update an edge and have one queued, we can return early.
	 * Otherwise, try to get a new pair of source and destination nodes. */
	if(op->r         != NULL  &&
	   op->edge_ctx  != NULL  &&
	   EdgeTraverseCtx_SetEdge(op->edge_ctx, op->r)) {
		return OpBase_CloneRecord(op->r);
	}

	NodeID src_id  = INVALID_ENTITY_ID;
	NodeID dest_id = INVALID_ENTITY_ID;

	while(true) {
		GrB_Info info = RG_MatrixTupleIter_next_UINT64(&op->iter, &src_id, &dest_id, NULL);

		// Managed to get a tuple, break.
		if(info == GrB_SUCCESS) break;

		/* Run out of tuples, try to get new data.
		 * Free old records. */
		op->r = NULL;
		for(uint i = 0; i < op->record_count; i++) OpBase_DeleteRecord(op->records[i]);

		// Ask child operations for data.
		for(op->record_count = 0; op->record_count < op->record_cap; op->record_count++) {
			Record childRecord = OpBase_Consume(child);
			// If the Record is NULL, the child has been depleted.
			if(!childRecord) break;
			if(!Record_GetNode(childRecord, op->srcNodeIdx)) {
				/* The child Record may not contain the source node in scenarios like
				 * a failed OPTIONAL MATCH. In this case, delete the Record and try again. */
				OpBase_DeleteRecord(childRecord);
				op->record_count--;
				continue;
			}

			// Store received record.
			Record_PersistScalars(childRecord);
			op->records[op->record_count] = childRecord;
		}

		// No data.
		if(op->record_count == 0) return NULL;

		_traverse(op);
	}

	/* Get node from current column. */
	op->r = op->records[src_id];
	// Populate the destination node and add it to the Record.
	Node destNode = GE_NEW_NODE();
	Graph_GetNode(op->graph, dest_id, &destNode);
	Record_AddNode(op->r, op->destNodeIdx, destNode);

	if(op->edge_ctx) {
		Node *srcNode = Record_GetNode(op->r, op->srcNodeIdx);
		// Collect all appropriate edges connecting the current pair of endpoints.
		EdgeTraverseCtx_CollectEdges(op->edge_ctx, ENTITY_GET_ID(srcNode), ENTITY_GET_ID(&destNode));
		// We're guaranteed to have at least one edge.
		EdgeTraverseCtx_SetEdge(op->edge_ctx, op->r);
	}

	return OpBase_CloneRecord(op->r);
}
```

