# LHLO Graph Optimization Passes

## 1 Fusion Pass

There are multiple fusion strategies, "base" and "stitch" for the current version.

"base" is an XLA style fusion strategy, aka, loop or input fusions supporting identical num-of-elements roots for multiple outputs. Something that requires additional consideration is that, when shapes are unknown, it's not straight forward to tell if num-of-elements of two tensors are identical. This is solved with the analysis of graph topology, taking the operation semantics into consideration.

"stitch" is a more aggressive fusion & codegen strategy, which allows multiple different loop/input schedules with different sizes to be fused together, by utilizing shared memories in GPU backend, or cache memories in CPU backend. "stitch" is the default fusion & codegen strategy for the CPU backend, and the GPU version is still ongoing for now, and thus is out of the scope of this
tutorial. 

- `DiscFusionPass`
- `DiscStitchFusionPass`

## 2 Speculation Pass

"speculation" pass creates multiple copies of an lmhlo.fusion, with different hint attributes guiding the kernel code to be generated with different strategies. 

- `DiscSpecializeFusionWithSpeculationPass`

For the current version, "speculation" happens on different dimensions, including:

* Implicit Broadcast. Both TensorFlow & PyTorch supports an implicit broadcast semantic, which is quite unfriendly to a dynamic shape compiler, since it's hard to tell if the broadcast is needed when the size is unknown. Usually,  a broadcast is not needed but if we cannot guarantee that in compile time and codegen with unnecessary broadcasts, the performance will suffer. So two versions of kernels will be generated, the original version and a simplified version. The simplified version will only be executed when the runtime shapes meet the conditions of no implicit broadcasts.

* Vectorization. Empirically, vectorization benefits when element_number % vectorize_size is 0. However, this can not be known at compile time. Thus a heuristic rule will generate two versions of kernels and only launch the vectorized one when the condition is met.

* Launch Dimension Selection. A heuristic rule to select different thread-per-blocks according to different sizes.

## 3 Memory Optimization Passes

The other two passes are related to memory optimization, which can also be regarded as part of bufferization.

- `PromoteBuffersToStackPass`

  reuse the building block in MLIR repo, promote the small CPU buffer allocation from memref.alloc to memref.alloca, which is regarded as intermediate buffers for shape calculation.

- `BufferDeallocationPass `

  insert the memref.dealloc op in the optimal position after the graph optimization on buffer level is done. This is actually the last step of bufferization.