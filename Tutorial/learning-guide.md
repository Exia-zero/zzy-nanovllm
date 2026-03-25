# nano-vllm 学习建议

这个仓库很适合学习 LLM 推理引擎，不适合从模型论文角度直接切入。更高效的方式是先抓住运行时主线：

`LLM.generate()` -> `Scheduler` -> `ModelRunner` -> `Attention / KV cache` -> `Sampler`

## 仓库定位

- 这是一个“用较少 Python 代码复现 vLLM 核心思路”的项目。
- 重点不是训练，而是推理时的请求调度、KV Cache 管理、Prefix Cache、Tensor Parallel、CUDA Graph 等机制。
- 代码量不大，适合建立对推理引擎的整体认知。

## 推荐阅读顺序

### 1. 从外部调用开始

先读根目录的示例文件：

- `example.py`

你需要先看清楚两件事：

- 这个项目对外暴露的 API 是什么
- 用户传入 prompt 后，内部大致会走哪条执行链

第一次学习建议固定使用：

- `tensor_parallel_size=1`
- `enforce_eager=True`

原因很简单：先把单卡、非图捕获路径看明白，再进入复杂优化。

### 2. 理解总入口和请求生命周期

然后读：

- `nanovllm/llm.py`
- `nanovllm/engine/llm_engine.py`

重点关注：

- `generate`
- `add_request`
- `step`

这里要建立一个最重要的全局认知：

- `generate()` 不是一次性跑完所有 token
- 它是在循环里不断调用 `step()`
- 每个 `step()` 都包含一次调度和一次模型执行

建议你边读边回答：

- prompt 在哪一步被 tokenize
- 请求对象是什么时候变成 `Sequence`
- 输出 token 是在哪一步回填到请求上的

### 3. 先彻底看懂调度器

然后读：

- `nanovllm/engine/sequence.py`
- `nanovllm/engine/scheduler.py`

这是学习推理引擎的核心，不是附属模块。

你要重点看懂：

- `Sequence` 维护了哪些状态
- `WAITING / RUNNING / FINISHED` 如何切换
- 调度器为什么区分 `prefill` 和 `decode`
- 为什么 `prefill` 优先按 token 数量受限
- 为什么 `decode` 更接近“每条序列每轮生成一个 token”

必须搞明白的概念：

- prefill：处理整段 prompt，把上下文一次性灌入 KV cache
- decode：每轮只生成一个 token，并继续使用已有 cache

如果这里没吃透，后面 `KV cache` 和 `paged attention` 会看得很碎。

### 4. 重点研究 BlockManager

然后读：

- `nanovllm/engine/block_manager.py`

这是整个仓库最值得反复看的文件之一。

它解决的是：

- KV cache 怎么分页
- block 怎么分配与释放
- prefix cache 怎么命中
- 显存不够时如何 preempt

读这一部分时重点盯住：

- `allocate`
- `deallocate`
- `can_append`
- `may_append`
- `compute_hash`

你要真正理解：

- 为什么 prefix cache 只对完整 block 生效
- 为什么 hash 要把前缀 block 的 hash 混进去
- 为什么 block 需要 `ref_count`
- 为什么 decode 过程中有时需要抢占已有序列

如果你能把 `block_table`、`num_cached_tokens`、`ref_count` 三者关系说清楚，说明这部分基本学会了。

### 5. 再看 ModelRunner，把调度和执行接起来

接着读：

- `nanovllm/engine/model_runner.py`

这个文件负责把调度结果变成真正的模型执行输入。

重点看：

- `prepare_prefill`
- `prepare_decode`
- `run_model`
- `allocate_kv_cache`
- `capture_cudagraph`

这里要搞懂几个运行时关键数据结构：

- `slot_mapping`
- `block_tables`
- `cu_seqlens_q`
- `cu_seqlens_k`
- `context_lens`

建议你带着下面的问题去读：

- `slot_mapping` 为什么本质上是在描述“token 应写入 KV cache 的物理位置”
- prefix cache 命中后，为什么 `q` 的长度和 `k/v` 的长度会不一样
- 为什么 prefill 和 decode 的输入组织方式完全不同
- 为什么 CUDA Graph 更适合 decode，而不是 prefill

这一层读懂后，你会第一次真正理解 vLLM 风格推理引擎和普通 `model.generate()` 的区别。

### 6. 最后看模型层如何配合 runtime

再读：

- `nanovllm/models/qwen3.py`
- `nanovllm/layers/attention.py`
- `nanovllm/layers/linear.py`
- `nanovllm/layers/embed_head.py`
- `nanovllm/layers/rotary_embedding.py`
- `nanovllm/layers/layernorm.py`
- `nanovllm/layers/sampler.py`

这部分不要以“又在看一个 Transformer 实现”的方式去看，而要盯住“模型层怎样配合推理引擎”。

尤其要看：

- `Attention` 如何读写 KV cache
- FlashAttention 在 prefill 和 decode 阶段分别怎么调用
- Tensor Parallel 是如何切分线性层和词表的
- `ParallelLMHead` 为什么在 prefill 只取每个请求最后一个位置的 hidden state

## 学习时最值得追问的问题

建议你把下面这些问题写在纸上，读代码时强制自己回答：

1. 为什么 `Scheduler` 要优先做 prefill？
2. 为什么 prefill 和 decode 的 batch 组织方式不同？
3. `Sequence` 为什么同时记录 `num_tokens`、`num_prompt_tokens`、`num_cached_tokens`？
4. `block_table` 存的是逻辑顺序还是物理地址？
5. 为什么 block 在完整填满前不能稳定参与 prefix cache？
6. decode 阶段为什么可能触发 preemption？
7. `slot_mapping` 和 `block_tables` 分别描述什么？
8. 为什么 `Attention` 在 prefix cache 场景下会直接使用 cache 中的 `k/v`？
9. CUDA Graph 为什么更适合固定形状、轻量重复的 decode？
10. Tensor Parallel 到底切的是权重、激活，还是两者都有？

如果这些问题你都能脱离代码解释清楚，这个仓库你就不只是“看过”，而是真的理解了。

## 最好的学习方式

### 1. 不要一开始就研究多卡

第一次阅读时，强烈建议：

- 只看单卡
- 只看 eager 模式
- 先不研究共享内存、多进程和 NCCL 细节

多卡路径会把注意力从核心机制上打散。

### 2. 给关键路径加日志

最有效的办法不是死读，而是自己加打印。

建议你在这些位置加日志：

- `Scheduler.schedule`
- `Scheduler.postprocess`
- `BlockManager.allocate`
- `BlockManager.may_append`
- `BlockManager.deallocate`
- `ModelRunner.prepare_prefill`
- `ModelRunner.prepare_decode`

建议打印：

- `seq_id`
- `len(seq)`
- `num_cached_tokens`
- `block_table`
- `slot_mapping`
- `context_lens`

只要你亲眼看到一轮 prefill、一轮 decode 里这些值怎么变化，理解速度会快很多。

### 3. 用“关闭一个优化”的方式学习

不要只看默认路径，最好做对照实验。

比如你可以分别体验：

- 关闭 prefix cache
- 打开/关闭 CUDA Graph
- 单卡和 Tensor Parallel 对比
- 短 prompt 和长 prompt 对比

性能差异会反过来逼你理解设计动机。

### 4. 做一个小改动来检验理解

最推荐的练习有两个：

- 给 `Sampler` 增加 greedy sampling
- 给 `Sampler` 增加 top-k 或 top-p

理由是这些改动范围小，但能逼你走完整条输出路径。

如果你想再进一层，可以尝试：

- 给 `Scheduler` 增加更明确的调试日志
- 给 `BlockManager` 增加统计信息，例如 cache hit rate

这种练习会让你真正掌握 runtime，而不是停留在“我知道它大概干了什么”。

## 一个实用的学习阶段划分

### 第一阶段：跑通

目标：

- 能跑起 `example.py`
- 知道输入输出长什么样
- 知道主入口在哪

成果标准：

- 你能从 `generate()` 讲到 `step()`

### 第二阶段：吃透调度和缓存

目标：

- 看懂 `Sequence`
- 看懂 `Scheduler`
- 看懂 `BlockManager`

成果标准：

- 你能解释 prefill/decode 的区别
- 你能解释 prefix cache 为什么成立

### 第三阶段：吃透执行层

目标：

- 看懂 `ModelRunner`
- 看懂 `Attention`
- 看懂 `slot_mapping` / `block_tables`

成果标准：

- 你能解释 token 是怎么被写入 paged KV cache 的

### 第四阶段：理解优化和扩展

目标：

- 看懂 CUDA Graph
- 看懂 Tensor Parallel
- 看懂模型权重加载和分片逻辑

成果标准：

- 你能解释这些优化为什么快，以及代价是什么

## 这个仓库的两个边界

先明确两个现实边界，避免学偏：

- 这是 CUDA/NCCL/FlashAttention/Triton 路线的实现，不是 CPU/MPS 通用推理教程。
- 这个仓库当前主要围绕 Qwen3 路径实现，所以你学到的核心是“vLLM 风格 runtime 机制”，不是完整多模型框架设计。

另外，根目录的 `test.py` 更像环境检查脚本，不是理解核心推理链的重点。

## 建议你的实际学习节奏

如果你想高效深入，建议按下面节奏走：

1. 跑 `example.py`，只关注外部调用。
2. 顺着 `LLM.generate()` 把主链读通。
3. 单独花一轮时间死磕 `Scheduler + BlockManager`。
4. 再花一轮时间死磕 `ModelRunner + Attention`。
5. 最后再看 Tensor Parallel、CUDA Graph、权重加载。
6. 每读完一层，就自己做一个小改动或加日志验证。

## 最终目标

你学这个仓库的目标不应该只是“会用它”，而应该是：

- 能解释一个现代 LLM 推理引擎是怎么组织请求的
- 能解释 paged KV cache 和 prefix cache 的核心机制
- 能说明 prefill 和 decode 为什么必须分开设计
- 能在这个仓库里独立做中小型改动

做到这一步，你再去读 vLLM 正式仓库，会轻松很多。
