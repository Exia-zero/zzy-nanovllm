# 第 2 课：环境、依赖与运行前提

## 本课在整套课程中的位置

第 1 课解决的是“这个项目是什么”，第 2 课解决的是“这个项目需要什么环境，依赖分别在系统里扮演什么角色”。

这节课很关键，因为很多人第一次学推理引擎时，容易把下面三件事混在一起：

- 项目声明了哪些依赖
- 代码实际依赖了哪些能力
- 自己当前机器能不能真的跑起来

这三件事如果不分开，后面读代码时会一直困惑，比如：

- 为什么 `test.py` 在测 MPS，但主执行路径明显是 CUDA
- 为什么 `pyproject.toml` 里有些依赖写得很清楚，有些实际用到的库却没列出来
- 为什么能导入项目，不代表能真正执行推理

所以这节课的核心任务，是把“依赖清单”和“运行前提”彻底拆开。

## 本课目标

学完这一课后，你应该能够：

1. 说清楚这个项目的核心依赖分别负责什么
2. 判断这个项目真正偏向哪条硬件路线
3. 区分“环境检查脚本”和“真实推理路径”的差异
4. 知道后续每个依赖会在哪些模块里出现

## 本课涉及文件

- `pyproject.toml`
- `test.py`
- `nanovllm/engine/model_runner.py`
- `nanovllm/layers/attention.py`
- `nanovllm/engine/block_manager.py`
- `nanovllm/utils/loader.py`

## 先给出结论

这一课你先记住三个结论：

1. 这个项目的声明依赖不等于它运行时真正需要的一切。
2. 这个项目的核心推理路径明显是 CUDA 路线，不是 MPS 路线。
3. `test.py` 更像是一个本地环境检查脚本，不是项目真实推理能力的权威说明。

## 一、先看 `pyproject.toml`：项目公开声明了什么

`pyproject.toml` 里声明了这些核心依赖：

- `torch>=2.4.0`
- `triton>=3.0.0`
- `transformers>=4.51.0`
- `flash-attn`
- `xxhash`

同时它要求：

- Python 版本：`>=3.10,<3.13`

这里你要先形成一个基本判断：

> 这个项目不是“纯 Python 教学项目”，而是明显依赖现代 GPU 推理栈的项目。

因为只要你看到下面这组组合：

- `torch`
- `triton`
- `flash-attn`

基本就应该立刻联想到：

- GPU 执行
- 高性能注意力
- 可能存在自定义 kernel
- 运行环境要求会高于一般 Python 项目

## 二、每个依赖在这个项目里到底干什么

这一部分是本课最核心的内容。

### 1. `torch`

`torch` 是整个项目的底座。

它在这个项目里承担了至少 6 类职责：

- 定义模型层和参数
- 管理张量和设备
- 提供 `torch.distributed`
- 提供 `torch.multiprocessing`
- 提供 `torch.compile`
- 提供 CUDA Graph 等能力

也就是说，`torch` 在这里不是简单的“深度学习库”，而是整个运行时系统的基础设施。

后面你会在这些地方频繁遇到它：

- `nanovllm/engine/model_runner.py`
- `nanovllm/models/qwen3.py`
- `nanovllm/layers/*.py`

### 2. `transformers`

`transformers` 在这个项目里主要承担两件事：

- 读取模型配置
- 提供 tokenizer

具体表现为：

- `config.py` 用 `AutoConfig`
- `engine/llm_engine.py` 和 `example.py` 用 `AutoTokenizer`
- `models/qwen3.py` 用 `Qwen3Config`

这说明项目没有自己造一整套模型配置生态，而是复用了 Hugging Face 的元数据和 tokenizer 工具链。

所以你要理解：

- nano-vllm 不是“替代 Transformers”
- 它是在 Transformers 生态之上，自建一套更靠近推理引擎的 runtime

### 3. `triton`

`triton` 只在一个地方出现，但信号量非常强：

- `nanovllm/layers/attention.py`

这里定义了 `store_kvcache_kernel`，负责把新的 `k/v` 写入 paged KV cache。

这说明：

- 项目不是只靠纯 PyTorch 组合操作
- 在关键路径上，作者愿意为了性能写定制 GPU kernel

但你要牢记一个学习顺序：

> Triton 很重要，但它不是现在最先该深挖的部分。

如果你还没看懂调度、block 管理、`slot_mapping`，先别钻 Triton kernel。

### 4. `flash-attn`

`flash-attn` 也出现在 `nanovllm/layers/attention.py`。

这里用到了两个关键函数：

- `flash_attn_varlen_func`
- `flash_attn_with_kvcache`

这意味着：

- prefill 和 decode 的注意力实现不是同一条路径
- 项目明显在围绕现代高性能注意力方案设计执行流程

你以后读 attention 部分时，要特别留意：

- 为什么 prefill 适合变长 attention
- 为什么 decode 更适合直接基于 kv cache 进行计算

### 5. `xxhash`

`xxhash` 出现在：

- `nanovllm/engine/block_manager.py`

它不是边角料，而是 prefix cache 的关键基础。

这个依赖一出现，你就应该意识到：

- 这里一定有哈希索引
- 哈希大概率和缓存命中判断有关

果然，后面你会看到 block 级前缀复用就是靠它做快速哈希判断。

### 6. `numpy`

虽然 `numpy` 没有在 `pyproject.toml` 里显式列出来，但 `block_manager.py` 里用了它：

- 用 `np.array(token_ids).tobytes()` 参与哈希计算

这里有一个学习点：

> 代码里实际依赖的东西，不一定都在最显眼的依赖列表里。

这不一定是严重问题，但说明你读项目时不能只看包管理文件，还要看代码真实 import 了什么。

### 7. `safetensors`

`utils/loader.py` 里还用了：

- `from safetensors import safe_open`

但它同样没有出现在 `pyproject.toml` 的依赖列表里。

这个现象很值得你记住，因为它告诉你：

- 这个项目的运行依赖，严格来说不只是一份 `pyproject.toml`
- 真正完整的环境要求，需要结合代码实际使用来判断

## 三、真实运行前提：这个项目明显是 CUDA 路线

只看 `test.py`，你可能会误以为：

- 这个项目主要面向 MPS
- 只要 Mac GPU 能跑矩阵乘法，就能跑 nano-vllm

这是不对的。

真正决定项目推理路径的不是 `test.py`，而是核心执行文件。

### 1. `model_runner.py` 直接说明了它依赖 CUDA

你在 `nanovllm/engine/model_runner.py` 里会看到：

- `dist.init_process_group("nccl", ...)`
- `torch.cuda.set_device(rank)`
- `torch.set_default_device("cuda")`
- `torch.cuda.mem_get_info()`
- `torch.cuda.CUDAGraph()`

这些 API 几乎已经明确写死了：

- CUDA
- NCCL
- GPU 显存管理
- CUDA Graph

所以从核心执行路径看，这个项目不是一条 MPS 主路径。

### 2. `flash-attn` 也进一步强化了 CUDA 依赖

`flash-attn` 在真实使用里通常意味着：

- 强依赖 CUDA 环境
- 对驱动、编译环境、GPU 架构都有要求

也就是说，哪怕 Python 依赖都装上了，也不等于项目一定能运行。

### 3. `triton` 自定义 kernel 也说明它不是“纯 CPU 项目”

`store_kvcache_kernel` 明确是 GPU kernel。

因此你应该把这个项目理解成：

- 逻辑上是 Python 项目
- 执行上是现代 GPU 推理项目

## 四、那为什么会有 `test.py`

`test.py` 的作用更像：

- 本地 PyTorch 安装是否正常
- 当前机器有没有 MPS 可用
- 当前解释器能不能进行基础张量运算

它能说明的是：

- 你的本地 Python 环境里 `torch` 可能是否正常

它不能说明的是：

- nano-vllm 的核心执行链能否在当前机器上跑通

所以这节课你要建立一个很重要的判断：

> 环境检测脚本，只能回答“部分环境是否正常”，不能直接代表项目完整运行条件。

## 五、从“依赖”到“文件”的映射关系

为了后续读代码更轻松，建议你现在就建立一张依赖映射表。

### `torch`

主要出现于：

- `nanovllm/engine/model_runner.py`
- `nanovllm/models/qwen3.py`
- `nanovllm/layers/linear.py`
- `nanovllm/layers/attention.py`
- `nanovllm/layers/layernorm.py`
- `nanovllm/layers/embed_head.py`

### `transformers`

主要出现于：

- `nanovllm/config.py`
- `nanovllm/engine/llm_engine.py`
- `example.py`
- `nanovllm/models/qwen3.py`

### `triton`

主要出现于：

- `nanovllm/layers/attention.py`

### `flash-attn`

主要出现于：

- `nanovllm/layers/attention.py`

### `xxhash`

主要出现于：

- `nanovllm/engine/block_manager.py`

### `safetensors`

主要出现于：

- `nanovllm/utils/loader.py`

这张映射表很有用，因为它帮你提前知道：

- 后面学哪个文件时，要把哪个依赖带着一起理解

### 一份可直接使用的依赖功能表

| 依赖名 | 声明类型 | 主要用途 | 关键出现文件 | 所属层 | 备注 |
| --- | --- | --- | --- | --- | --- |
| `torch` | 显式声明 | 张量计算、模型定义、设备管理、分布式通信、进程管理、`torch.compile`、CUDA Graph | `nanovllm/engine/model_runner.py`、`nanovllm/models/qwen3.py`、`nanovllm/layers/*.py` | 模型与算子层、执行准备层、底层加速层 | 整个项目的基础设施，没有它项目无法运行 |
| `transformers` | 显式声明 | 读取模型配置、提供 tokenizer、承接 Qwen3 配置对象 | `nanovllm/config.py`、`nanovllm/engine/llm_engine.py`、`example.py`、`nanovllm/models/qwen3.py` | 用户接口层、执行准备层 | nano-vllm 复用它的生态，不自己实现 tokenizer/config 体系 |
| `triton` | 显式声明 | 编写自定义 GPU kernel，把新生成的 `k/v` 写入 paged KV cache | `nanovllm/layers/attention.py` | 底层加速层 | 主要用于关键路径优化，不是课程最先要深挖的部分 |
| `flash-attn` | 显式声明 | 提供高性能注意力实现，分别覆盖 prefill 和 decode 路径 | `nanovllm/layers/attention.py` | 底层加速层 | 强烈暗示项目主路径是 CUDA，而不是 CPU/MPS |
| `xxhash` | 显式声明 | 为 block 级 prefix cache 提供快速哈希 | `nanovllm/engine/block_manager.py` | 运行时组织层 | 用于判断前缀 block 是否可以复用 |
| `numpy` | 隐含依赖 | 把 token 序列转成稳定字节串，参与哈希计算 | `nanovllm/engine/block_manager.py` | 运行时组织层 | 没写在 `pyproject.toml`，但代码实际使用了 |
| `safetensors` | 隐含依赖 | 高效读取模型权重文件 | `nanovllm/utils/loader.py` | 执行准备层 | 没写在 `pyproject.toml`，但模型加载依赖它 |
| `tqdm` | 隐含依赖 | 生成过程中的进度条和吞吐展示 | `nanovllm/engine/llm_engine.py` | 用户接口层 | 不影响核心算法，但影响交互体验 |
| `multiprocessing.shared_memory` | 标准库 | 多进程 tensor parallel 场景下传递调用信息 | `nanovllm/engine/model_runner.py` | 执行准备层 | 属于 Python 标准库，不需要额外安装 |
| `torch.distributed / NCCL` | `torch` 子能力 | 多卡通信、张量并行协作 | `nanovllm/engine/model_runner.py`、`nanovllm/layers/linear.py`、`nanovllm/layers/embed_head.py` | 执行准备层、底层加速层 | 真正多卡可用还依赖 NCCL 和 CUDA 环境 |

你可以把这张表当成第 2 课的标准答案版本来用。最关键的不是背表，而是从表里看出 3 件事：

- 哪些依赖属于“显式声明”
- 哪些依赖属于“代码隐含要求”
- 哪些依赖真正决定了项目的运行路线

## 六、从 Python 到 GPU 内核的执行栈

这是本课最建议你画出来的一张图。

可以把这个项目的执行栈理解为 5 层：

### 第 1 层：课程和用户接口层

- `example.py`
- `LLM`
- `SamplingParams`

这一层负责“怎么用”。

### 第 2 层：运行时组织层

- `LLMEngine`
- `Scheduler`
- `Sequence`
- `BlockManager`

这一层负责“请求怎么被组织和调度”。

### 第 3 层：执行准备层

- `ModelRunner`
- `Context`
- 权重加载

这一层负责“把调度结果变成模型可执行输入”。

### 第 4 层：模型与算子层

- `Qwen3ForCausalLM`
- `Attention`
- `Linear`
- `RMSNorm`
- `Sampler`

这一层负责“真正算出下一个 token”。

### 第 5 层：底层加速层

- CUDA
- NCCL
- FlashAttention
- Triton kernel

这一层负责“怎么更快”。

你如果能把这 5 层说顺，后面基本不会迷路。

## 七、本课最容易出现的误解

### 误解 1：`test.py` 能跑，项目就能跑

不对。

`test.py` 只是局部环境检查，不代表：

- `flash-attn` 可用
- CUDA 路径可用
- NCCL 可用
- 模型加载可用

### 误解 2：`pyproject.toml` 就是完整依赖真相

不完全对。

因为代码里还显式用了：

- `safetensors`
- `numpy`

所以你真正判断一个项目环境时，要看：

- 包管理文件
- 实际 import
- 核心执行路径

### 误解 3：有 PyTorch 就够了

不对。

这个项目依赖的不只是 PyTorch 本身，而是：

- 分布式通信
- CUDA 设备能力
- FlashAttention
- Triton

### 误解 4：MPS 是这个项目的主要路线

不是。

从核心文件看，项目主路径明显是 CUDA。

## 八、建议的环境认知清单

在进入第 3 课前，建议你能回答下面这些问题。

1. 这个项目对 Python 版本的要求是什么？
2. `torch` 在这个项目里承担了哪些职责？
3. 为什么 `flash-attn` 会强烈暗示 CUDA 路线？
4. `xxhash` 出现意味着后面大概率会学到什么？
5. 为什么 `test.py` 不足以代表项目完整运行能力？
6. 为什么说真实运行前提应该看 `model_runner.py` 而不是只看安装脚本？

## 九、本课建议动作

这一课建议你做三件事。

### 动作 1：自己做一份依赖功能表

格式可以非常简单：

- 依赖名
- 用途
- 出现文件
- 属于哪一层

### 动作 2：区分“声明依赖”和“隐含依赖”

你至少要把下面两类分开写：

- `pyproject.toml` 里明确写出的
- 代码实际 import 但没有显式声明的

### 动作 3：写出你的运行环境判断

请尝试写一句话回答：

> 我的当前机器，是更接近这个项目的目标环境，还是更接近一个只能读代码、不适合真实运行的环境？

这句话会强迫你把“能学”和“能跑”分开。

## 十、本课作业

### 作业 1

画一张 “从 Python 到 GPU 内核” 的依赖关系图，至少包含：

- Python
- torch
- transformers
- flash-attn
- triton
- xxhash
- safetensors

### 作业 2

用不超过 250 字回答：

> 为什么说 nano-vllm 的核心执行路径是 CUDA 路线，而不是 MPS 路线？

### 作业 3

列出 3 条你认为“安装成功了，但项目未必能跑”的原因。

## 十一、本课最终收获

学完这一课，你得到的不是“记住几个库名”，而是三种非常重要的工程判断能力：

- 看依赖时，知道哪些是主角，哪些是配角
- 看环境时，知道应该信谁：信核心执行路径，而不是信孤立脚本
- 看项目时，知道“能 import”和“能高性能运行”根本不是一回事

这三点会直接决定你后面读 `Scheduler`、`ModelRunner`、`Attention` 时是否清醒。

## 下一课预告

第 3 课会正式进入外部 API，主题是：

- 从示例程序认识 `LLM`
- `SamplingParams` 到底控制了什么
- tokenizer 和 chat template 如何进入系统

到那时，你就会从“环境准备”正式走进“执行主线”。
