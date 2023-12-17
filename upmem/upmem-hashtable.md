### upmem设计架构

  - 20 DIMMs.

    Intel server带有20个DIMM，一个DIMM是一个UPMEM 模块。

  - Each UPMEM PIM DIMM contains 128 DPUs

    每一个UPMEM模块包含128个DPU，也就是128个处理器。

  - Each DPU contains a 64-MB MRAM 、a 24-KB IRAM and a 64-KB WRAM

    每一个DPU中包含24-KB的 IRAM，64-KB的WRAM和64-MB的MRAM。IRAM为指令缓存；WRAM是程序的执行存储器。这是根据预定义方案放置堆栈、全局变量等的地方；MRAM被视为“外部外围设备”，对MRAM变量的每次访问都意味着MRAM传输。

  - The DPU is a 24 threads, 32-bit RISC processor

    DPU中的处理器为24线程的处理器，每个DPU拥有24个可用线程。

    <img src="https://user-images.githubusercontent.com/83715643/190892190-6d244cdf-c896-4f8b-914a-11fa98649526.png" alt="upmem_PIM_chip" style="zoom:50%;" />

### upmem编程

请详细阅读[upmem官网文档](https://sdk.upmem.com/)。

### upmem-hashtable设计

  - 总体架构

    <img src="https://user-images.githubusercontent.com/83715643/190892102-a16d525d-a064-4733-82b7-bbdc7688f98b.png" style="zoom:50%;" />
    
    - 将buckets分区存储在DPU中
    
      我们将buckets分区存储在各个DPU中。DPU0存储buckets[0] ~buckets[HT_SIZE_PER_PDU-1]，DPU1存储buckets[HT_SIZE_PER_PDU] ~buckets[2*HT_SIZE_PER_PDU-1]……
    
    - Host cache中存储元数据
    
      Host DRAM中存储Hashtale的一些元数据，比如，分配的DPU数量，每个DPU中存储的bucket数量。通过使用HashFunction将需要处理的请求分发至各个DPU。
    
    - 每次处理一个batch的请求
    
      因为host调用DPU处理请求会涉及系统调用，开销较大，我们选择每次处理一个batch的请求。
    
  - 增删改查的流程
  
    Hashtable收到请求后，首先，使用HashFunction计算出bucket_index；通过bucket_index和元数据计算出该请求所对应的DPU和bucket_id；将请求内容传输至对应的DPU；运行DPU，DPU解析请求，进行增删查改操作；最后，Host获取DPU处理结果。
    
    下面为增删查改流程的伪代码：
    
    ```c
    index = HashFunction(key);
    dpu = index / HT_SIZE_PER_PDU; // 获取操作dpu_id
    bucket = index % HT_SIZE_PER_PDU; // 获取对应dpu中bucket_id
    dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, request); // 传输请求内容
    dpu_launch(dpu_set, DPU_SYNCHRONOUS); // 运行dpu
    dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, response); // 获取response
    ```
    
  - MRAM_allocator
  
    MRAM_allocator为MRAM设计的一个线性内存分配器。
  
    <img src="https://user-images.githubusercontent.com/83715643/190892252-5ea94b93-a2e2-4011-b19a-b0841a81db44.png" alt="MRAM" style="zoom:50%;" />
    
    - DPU 程序可以自由使用从MRAM`DPU_MRAM_HEAP_POINTER`到末尾的范围。
    
      UPMEM DPU运行库对MRAM进行了内存管理：MRAM分为了两部分，0到PU_MRAM_HEAP_POINTER为MRAM变量使用，大小为1MB；DPU_MRAM_HEAP_POINTER到末尾的部分由DPU程序自由使用，大小为63MB。
    
    - 我们设计了线性内存分配器对MRAM中自由支配的部分进行管理。
    
      DPU_MRAM_HEAP_POINTER到末尾的部分可以视为裸内存，我们设计了一个线性内存分配器对其进行管理。
    
    - 内存分配需要满足8字节对齐，以便于DMA传输。

### upmem-hashtable源码

代码已上传至[github](https://github.com/SJTU-DDST/hashtable-upmem)。

### upmem-hashtable优化

我们提出了一些优化意见，在后续的upmem-hashtable版本中，我们会实现这些优化。

  - 并发
    - 每个DPU可用24线程，需要12以上线程才能拉满性能
    - 对于查询操作，直接并发查询
    - 对于增删改操作，需要同步加锁。
  - 避免DPU-CPU频繁交互
    - 使用`dpu_push_xfer()`并行传输，可提升传输效率。
    - `dpu_push_xfer()`和`dpu_launch()`均涉及系统调用，开销很大
    - 每次处理一个batch的request
  - MRAM_allocator
    - 设计动态内存分配器
  - WRAM和MRAM之间的数据传输
    - 对于较大的字符串，采用`mram_write()`和`mram_read()`提升传输效率。