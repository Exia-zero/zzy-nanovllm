# nano-vllm 30 节课程大纲

## 课程目标

这套课程面向“想系统理解 nano-vllm 以及 vLLM 风格推理引擎核心机制”的学习者，目标不是只会运行项目，而是能够：

- 理解从 `LLM.generate()` 到 token 输出的完整执行链
- 理解 `prefill / decode` 的区别与设计动机
- 理解 paged KV cache、prefix cache、调度与抢占机制
- 理解模型执行层如何和运行时协作
- 理解 Tensor Parallel、CUDA Graph、权重加载等优化路径
- 具备在这个仓库上做中小型改动的能力

## 课程设计原则

- 先整体，后细节
- 先单卡 eager，再多卡和优化
- 先调度与缓存，再模型层
- 每节课都尽量带一个观察任务或动手练习

## 课程结构

- 第一阶段：入门与运行主线，1-6 课
- 第二阶段：请求生命周期与调度，7-12 课
- 第三阶段：KV cache 与执行上下文，13-18 课
- 第四阶段：模型执行与注意力实现，19-24 课
- 第五阶段：并行、加载与性能优化，25-28 课
- 第六阶段：验证、扩展与综合实践，29-30 课

---

## 第一阶段：入门与运行主线

### 第 1 课：项目定位与学习地图

- 目标：明确这个仓库解决什么问题，不解决什么问题。
- 重点：理解它是推理引擎学习项目，不是完整训练框架。
- 文件：`README.md`，`pyproject.toml`
- 练习：用自己的话写出“nano-vllm 和普通 Hugging Face `generate()` 的区别”。

### 第 2 课：环境、依赖与运行前提

- 目标：搞清楚项目依赖栈和硬件前提。
- 重点：`torch`、`triton`、`flash-attn`、`transformers`、`xxhash` 分别承担什么角色。
- 文件：`pyproject.toml`，`test.py`
- 练习：画一张“从 Python 到 GPU 内核”的依赖关系图。

### 第 3 课：从示例程序认识外部 API

- 目标：从用户视角理解最小使用方式。
- 重点：`LLM`、`SamplingParams`、tokenizer、chat template 的关系。
- 文件：`example.py`
- 练习：修改 prompt，分别输入普通文本和 chat template 文本，观察输出差异。

### 第 4 课：配置对象与运行边界

- 目标：理解所有全局配置如何进入系统。
- 重点：`max_num_batched_tokens`、`max_num_seqs`、`max_model_len`、`tensor_parallel_size`、`enforce_eager`。
- 文件：`nanovllm/config.py`，`nanovllm/sampling_params.py`
- 练习：解释为什么 `max_num_batched_tokens >= max_model_len` 必须成立。

### 第 5 课：总入口 `LLM` 与 `LLMEngine`

- 目标：看懂最上层抽象。
- 重点：`LLM` 几乎只是别名，核心逻辑在 `LLMEngine`。
- 文件：`nanovllm/llm.py`，`nanovllm/engine/llm_engine.py`
- 练习：写出 `generate()` 内部依次调用了哪些核心步骤。

### 第 6 课：第一次走通完整执行链

- 目标：站在全局视角看一次完整推理流程。
- 重点：`add_request`、`step`、`schedule`、`run`、`postprocess` 的连接关系。
- 文件：`nanovllm/engine/llm_engine.py`
- 练习：画一张从 prompt 输入到 completion 输出的时序图。

---

## 第二阶段：请求生命周期与调度

### 第 7 课：`Sequence` 数据结构设计

- 目标：理解一个请求在系统里的最小表示。
- 重点：`seq_id`、`status`、`token_ids`、`num_prompt_tokens`、`num_cached_tokens`、`block_table`。
- 文件：`nanovllm/engine/sequence.py`
- 练习：解释为什么 `Sequence` 同时保留 `last_token` 和完整 token 序列。

### 第 8 课：请求状态机

- 目标：理解 `WAITING / RUNNING / FINISHED` 的切换逻辑。
- 重点：状态如何驱动调度和资源管理。
- 文件：`nanovllm/engine/sequence.py`，`nanovllm/engine/scheduler.py`
- 练习：列出一个请求从进入队列到结束的全部状态变化。

### 第 9 课：为什么要区分 `prefill` 和 `decode`

- 目标：建立推理引擎最重要的两个阶段概念。
- 重点：prompt 批量处理与逐 token 生成的本质差异。
- 文件：`nanovllm/engine/scheduler.py`，`nanovllm/engine/model_runner.py`
- 练习：用一句话解释为什么这两个阶段不能用同一种 batch 组织方式。

### 第 10 课：调度器的 prefill 逻辑

- 目标：看懂请求首次进入执行队列时发生了什么。
- 重点：`waiting` 队列、`max_num_seqs`、`max_num_batched_tokens`、`can_allocate`。
- 文件：`nanovllm/engine/scheduler.py`
- 练习：自己推演 3 个不同长度请求被 prefill 打包时的结果。

### 第 11 课：调度器的 decode 逻辑

- 目标：看懂进入运行期后的持续调度。
- 重点：`can_append`、`may_append`、一轮一个 token 的 decode 节奏。
- 文件：`nanovllm/engine/scheduler.py`
- 练习：解释为什么 decode 阶段通常每条序列每轮只前进一步。

### 第 12 课：抢占与恢复

- 目标：理解显存压力下的 preemption 机制。
- 重点：为什么需要把运行中的序列重新放回 `waiting`。
- 文件：`nanovllm/engine/scheduler.py`，`nanovllm/engine/block_manager.py`
- 练习：举例说明如果没有抢占机制，会在哪种场景卡死。

---

## 第三阶段：KV cache 与执行上下文

### 第 13 课：为什么推理引擎离不开 KV cache

- 目标：先从概念上理解 cache 的必要性。
- 重点：避免重复计算历史 token 的 `k/v`。
- 文件：无，概念课
- 练习：对比“无缓存自回归生成”和“有缓存生成”的复杂度差异。

### 第 14 课：Block 抽象与分页思路

- 目标：理解为什么 KV cache 要按 block 管理。
- 重点：逻辑序列与物理 cache 分页之间的映射。
- 文件：`nanovllm/engine/block_manager.py`
- 练习：解释 `block_table` 存储的是哪一类信息。

### 第 15 课：`BlockManager.allocate` 的工作原理

- 目标：看懂首轮分配时的行为。
- 重点：完整 block、部分 block、cache miss、cache hit。
- 文件：`nanovllm/engine/block_manager.py`
- 练习：手工推演一个长 prompt 被切成多个 block 的过程。

### 第 16 课：Prefix Cache 的哈希设计

- 目标：理解 prefix cache 命中的条件。
- 重点：`compute_hash`、前缀哈希拼接、完整 block 才可稳定复用。
- 文件：`nanovllm/engine/block_manager.py`
- 练习：解释为什么最后一个未满 block 不能稳定进入 prefix cache。

### 第 17 课：Block 生命周期与引用计数

- 目标：理解 block 为什么需要 `ref_count`。
- 重点：复用、释放、共享与去重。
- 文件：`nanovllm/engine/block_manager.py`
- 练习：举例说明两个请求共享前缀时 `ref_count` 如何变化。

### 第 18 课：执行上下文对象

- 目标：理解运行时是怎样把调度信息传给注意力层的。
- 重点：`Context`、`slot_mapping`、`block_tables`、`context_lens`、`cu_seqlens_q`、`cu_seqlens_k`。
- 文件：`nanovllm/utils/context.py`，`nanovllm/engine/model_runner.py`
- 练习：区分每个字段是用于 prefill 还是 decode。

---

## 第四阶段：模型执行与注意力实现

### 第 19 课：`ModelRunner` 总体职责

- 目标：把调度层和模型层真正接起来。
- 重点：进程初始化、模型构建、采样器、warmup、kv cache 分配。
- 文件：`nanovllm/engine/model_runner.py`
- 练习：总结 `ModelRunner` 在系统中承担的 5 个职责。

### 第 20 课：显存测量与 KV Cache 分配

- 目标：理解 `allocate_kv_cache` 如何估算可用 block 数量。
- 重点：模型权重、峰值显存、当前显存、`gpu_memory_utilization`。
- 文件：`nanovllm/engine/model_runner.py`
- 练习：解释 `num_kvcache_blocks` 的计算逻辑。

### 第 21 课：Prefill 输入准备

- 目标：看懂 `prepare_prefill`。
- 重点：为什么要构造 `input_ids`、`positions`、`cu_seqlens_q/k`、`slot_mapping`。
- 文件：`nanovllm/engine/model_runner.py`
- 练习：推演一个 prefix cache 命中场景下 `q` 和 `k/v` 长度如何不同。

### 第 22 课：Decode 输入准备

- 目标：看懂 `prepare_decode`。
- 重点：为什么 decode 只输入 `last_token`，以及 `context_lens` 和 `block_tables` 的作用。
- 文件：`nanovllm/engine/model_runner.py`
- 练习：解释为什么 decode 的 batch 形状比 prefill 更稳定。

### 第 23 课：注意力层中的 KV Cache 写入

- 目标：理解 token 是如何被写入 paged KV cache 的。
- 重点：`store_kvcache_kernel`、`slot_mapping` 到物理位置的映射。
- 文件：`nanovllm/layers/attention.py`
- 练习：用自己的话说明这个 Triton kernel 实际写入了什么。

### 第 24 课：FlashAttention 的 prefill 与 decode 两条路径

- 目标：理解两种注意力调用方式。
- 重点：`flash_attn_varlen_func` 与 `flash_attn_with_kvcache` 的区别。
- 文件：`nanovllm/layers/attention.py`
- 练习：解释 prefix cache 命中时，为什么注意力层可以直接读 cache 中的 `k/v`。

---

## 第五阶段：模型结构、并行与性能优化

### 第 25 课：Qwen3 模型结构在这个仓库里的最小实现

- 目标：理解模型层并不是重点，但必须能读懂。
- 重点：Embedding、DecoderLayer、Attention、MLP、Norm、LM Head。
- 文件：`nanovllm/models/qwen3.py`
- 练习：画出 `Qwen3ForCausalLM` 的前向路径。

### 第 26 课：RoPE、RMSNorm、MLP 与采样器

- 目标：理解辅助层如何为推理服务。
- 重点：`RotaryEmbedding`、`RMSNorm`、`SiluAndMul`、`Sampler`。
- 文件：`nanovllm/layers/rotary_embedding.py`，`nanovllm/layers/layernorm.py`，`nanovllm/layers/activation.py`，`nanovllm/layers/sampler.py`
- 练习：给 `Sampler` 增加 greedy sampling 支持。

### 第 27 课：Tensor Parallel 在线性层中的实现

- 目标：理解张量并行如何切分权重。
- 重点：`ColumnParallelLinear`、`RowParallelLinear`、`QKVParallelLinear`、`MergedColumnParallelLinear`。
- 文件：`nanovllm/layers/linear.py`
- 练习：解释为什么有的层切输出维，有的层切输入维。

### 第 28 课：词表并行、权重加载与 CUDA Graph

- 目标：把并行与优化路径补全。
- 重点：`VocabParallelEmbedding`、`ParallelLMHead`、packed 权重加载、`capture_cudagraph`。
- 文件：`nanovllm/layers/embed_head.py`，`nanovllm/utils/loader.py`，`nanovllm/engine/model_runner.py`
- 练习：解释为什么 CUDA Graph 更适合 decode 阶段。

---

## 第六阶段：验证、扩展与综合实践

### 第 29 课：基准测试、日志与调试方法

- 目标：学会验证自己的理解，而不是只读代码。
- 重点：`bench.py`、关键路径日志、cache hit rate、吞吐观测。
- 文件：`bench.py`，`nanovllm/engine/scheduler.py`，`nanovllm/engine/block_manager.py`
- 练习：加日志打印 `seq_id`、`num_cached_tokens`、`block_table` 并解释输出。

### 第 30 课：综合实践与扩展项目

- 目标：通过一个完整改动检验是否真正掌握。
- 可选项目：
  - 增加 `top-k` 或 `top-p` 采样
  - 给 `BlockManager` 增加 prefix cache 命中统计
  - 给 `Scheduler` 增加更清晰的调试输出
  - 尝试支持一个新的模型结构
- 成果标准：
  - 能解释改动影响了哪条执行链
  - 能说明改动前后功能和性能的变化

---

## 推荐学习节奏

如果你按周推进，这 30 节课可以拆成 6 周：

- 第 1 周：1-6 课，建立主线认知
- 第 2 周：7-12 课，彻底吃透调度
- 第 3 周：13-18 课，彻底吃透 KV cache 和上下文
- 第 4 周：19-24 课，搞懂执行层和注意力
- 第 5 周：25-28 课，理解模型结构、并行与优化
- 第 6 周：29-30 课，做验证和综合改动

## 学完后的能力标准

完成这 30 节课后，理想状态下你应该能够：

- 独立讲清楚 `generate()` 到 token 输出的全链路
- 独立解释 paged KV cache、prefix cache、preemption 的核心逻辑
- 解释为什么 `prefill / decode` 必须分开设计
- 解释 Tensor Parallel 与 CUDA Graph 在这个项目里的落地方式
- 在这个仓库上做一项中等复杂度的功能扩展
