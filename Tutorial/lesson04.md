# 第 4 课：配置对象与运行边界

## 本课在整套课程中的位置

第 3 课你已经从 `example.py` 看到了外部 API 长什么样。现在要进一步回答：

> 用户传给系统的参数，到底分成哪几类？它们各自约束了什么？

这就是第 4 课的主题。

这一课的目标不是记住几个字段名，而是建立一个非常重要的工程意识：

- 哪些参数决定“引擎怎么建”
- 哪些参数决定“请求怎么采样”
- 哪些信息是系统运行后才推导出来的

这三类信息如果混在一起，后面读 `LLMEngine`、`Scheduler`、`ModelRunner` 会非常乱。

## 本课目标

学完这一课后，你应该能够：

1. 区分 `Config` 和 `SamplingParams` 的职责
2. 说清楚每个关键配置项在限制什么
3. 理解为什么 `Config` 里有些字段是用户传入，有些字段是运行后推导
4. 说清楚“运行边界”这个词在这个项目里具体指什么

## 本课涉及文件

- `nanovllm/config.py`
- `nanovllm/sampling_params.py`
- `nanovllm/engine/llm_engine.py`

## 先给出结论

这一课你先记住 4 个结论：

1. `Config` 负责描述引擎级配置，决定这个运行时实例怎么被构造
2. `SamplingParams` 负责描述请求级采样策略，决定每次生成怎么停止和怎么采样
3. `Config` 里有一部分字段来自用户输入，另一部分来自模型配置或运行时推导
4. 所谓“运行边界”，就是系统能处理多长、多少条、多少显存、多大并行度的限制集合

## 一、为什么这个项目要把配置拆成两个对象

先看第 3 课里的示例：

- `LLM(path, enforce_eager=True, tensor_parallel_size=1)`
- `SamplingParams(temperature=0.6, max_tokens=256)`

这里已经隐含了一种很清晰的分层思想。

### 1. 一类参数属于引擎

比如：

- 模型路径
- 最大 batch token 数
- 并行度
- 是否强制 eager

这些参数一旦确定，整个运行时实例的行为边界就基本确定了。

### 2. 一类参数属于请求

比如：

- 温度
- 最多生成多少 token
- 是否忽略 EOS

这些参数不应该和引擎实例绑定死，因为不同请求可以有不同采样需求。

所以把配置拆成两个对象，不只是代码风格问题，而是在表达：

> 系统级约束和请求级策略必须分离。

## 二、先看 `Config`：它描述了什么

`Config` 是一个 dataclass。

这说明作者想要的是：

- 字段清晰
- 初始化集中
- 便于从 `kwargs` 中筛选

`Config` 的字段可以分成 5 类。

## 三、第一类字段：模型与路径

最核心的是：

- `model: str`

这不是一个普通参数，它是整个系统的根输入。

它决定：

- 去哪里读 `AutoConfig`
- 去哪里读 tokenizer
- 去哪里读模型权重

所以 `model` 字段的真正含义是：

> 运行时实例依附于哪一个本地模型目录。

## 四、第二类字段：batch 与并发边界

这一类包括：

- `max_num_batched_tokens`
- `max_num_seqs`
- `max_model_len`

它们共同决定系统一次最多处理多少工作量。

### 1. `max_num_batched_tokens`

它约束的是：

- 一次调度里最多能塞多少 token 工作量

你后面学 `Scheduler` 时会发现，这个值对 prefill 特别重要，因为 prefill 阶段是按 token 工作量组织 batch 的。

### 2. `max_num_seqs`

它约束的是：

- 一次最多同时处理多少条序列

这个值会影响：

- 调度器
- CUDA Graph 批尺寸集合
- 总体并发上限

### 3. `max_model_len`

它约束的是：

- 单条请求允许的最大上下文长度

这个值不会无条件取用户输入，而是会和模型本身支持的上限取最小值。

这说明作者的思路很工程化：

- 用户可以提需求
- 系统要用模型真实能力校正这个需求

## 五、第三类字段：显存与执行策略

这一类包括：

- `gpu_memory_utilization`
- `tensor_parallel_size`
- `enforce_eager`

### 1. `gpu_memory_utilization`

这个值不是表面上那么简单。

它实际上会影响：

- 预留多少显存给 KV cache
- 最终能分配多少个 cache blocks

也就是说，它不是一个“性能调味参数”，而是后面 cache 容量的根参数之一。

### 2. `tensor_parallel_size`

这个字段决定：

- 需要启动几个 rank
- 线性层和词表分片怎么进行
- 是否进入多进程通信路径

所以它不是“优化开关”那么简单，它决定了系统结构本身是否进入多卡模式。

### 3. `enforce_eager`

这个字段决定：

- 是否强制走 eager 执行
- 是否跳过 CUDA Graph 捕获

从学习角度看，这个参数非常友好，因为它允许你先走最直接、最好理解的执行路径。

## 六、第四类字段：由模型或运行时推导出来的信息

这一类字段更有工程意义：

- `hf_config`
- `eos`
- `num_kvcache_blocks`

这些字段虽然放在 `Config` 里，但并不是用户直接传进来的业务参数。

### 1. `hf_config`

它来自：

- `AutoConfig.from_pretrained(self.model)`

它的作用是：

- 让系统拿到模型结构元信息

比如后面会用到：

- hidden size
- attention heads
- max position embeddings

### 2. `eos`

它后面会由 tokenizer 的 `eos_token_id` 回填。

也就是说：

- `Config` 会先被创建
- 然后再补上模型和 tokenizer 推导出来的信息

### 3. `num_kvcache_blocks`

它初始时是 `-1`，但后面会由运行时根据显存情况推导出来。

这告诉你一件事：

> `Config` 不只是“用户输入快照”，它还承担了一部分运行时状态承载功能。

## 七、第五类字段：KV cache 几何参数

这一类包括：

- `kvcache_block_size`
- `num_kvcache_blocks`

其中 `kvcache_block_size=256` 非常关键，因为它会直接影响：

- block 切分粒度
- prefix cache 命中粒度
- block table 的长度

你后面学 `BlockManager` 时会反复回到这个字段。

所以现在先记住：

- block size 不是无关紧要的小参数
- 它是整个 paged KV cache 设计的基础几何单位

## 八、`__post_init__` 里做了什么

`Config.__post_init__()` 是本课最值得细读的一段。

它做了几件事情：

### 1. 检查模型路径必须存在

这说明系统假设：

- 运行前模型目录已经准备好

### 2. 检查 block size 必须是 256 的倍数

这说明：

- block 粒度并不是任意可选
- 后续实现对它有明确假设

### 3. 检查并行度范围

- `1 <= tensor_parallel_size <= 8`

说明这不是一个完全开放的实验参数，而是作者当前实现愿意支持的范围。

### 4. 读取 Hugging Face 模型配置

这一步把用户给的模型路径，转换成了系统真正能理解的结构信息。

### 5. 用模型能力修正 `max_model_len`

这里非常重要：

- 不是用户写多少就用多少
- 而是 `min(user_setting, model_limit)`

这体现出运行边界的真实含义：

> 系统边界不仅由用户配置决定，还由模型物理能力决定。

### 6. 确保 `max_num_batched_tokens >= max_model_len`

这个断言非常有教学意义。

它等于在说：

- 如果一次 batch 能处理的 token 总数，甚至小于单条序列允许的最大长度
- 那系统的调度逻辑会很别扭，甚至不成立

所以这里不是随便写了个 assert，而是在保护调度前提。

## 九、再看 `SamplingParams`

`SamplingParams` 很短，但作用非常集中。

它只有 3 个字段：

- `temperature`
- `max_tokens`
- `ignore_eos`

你应该把它理解为：

> 这是一次生成请求的结束条件和采样风格描述。

### 1. `temperature`

控制采样随机性。

这个项目里有个很值得注意的设计：

- `temperature > 1e-10`

也就是说，它当前不允许真正的 greedy 模式。

这件事不是偶然，它告诉你：

- 当前项目实现重点不在完整采样策略覆盖
- 后面把 greedy sampling 当练习题很合适

### 2. `max_tokens`

控制：

- 最多生成多少 completion token

这会直接影响请求何时结束。

### 3. `ignore_eos`

控制：

- 如果遇到 `eos_token`，是否仍然继续生成

这个字段后面会直接参与结束判断。

## 十、`Config` 和 `SamplingParams` 的真正区别

建议你用一句最短的话把它们分开：

- `Config` 决定引擎“长什么样”
- `SamplingParams` 决定请求“怎么生成”

再展开一点：

### `Config`

更像：

- 系统构造参数
- 资源边界
- 执行策略
- 模型能力约束

### `SamplingParams`

更像：

- 请求级行为控制
- 采样方式
- 结束条件

如果你把这两类信息混在一起，后面就会理解错很多设计。

## 十一、什么叫“运行边界”

这一课反复提这个词，现在把它讲清楚。

在这个项目里，运行边界主要是下面这些约束的总和：

- 模型目录是否存在
- 模型本身支持多长上下文
- 系统一次最多处理多少 token
- 一次最多处理多少条序列
- GPU 显存允许切出多少 KV cache
- 是否多卡
- 是否允许图捕获优化

所以“运行边界”不是一个参数，而是一组共同约束。

后面你学习系统行为时，很多“为什么不能这么做”，答案其实都在这些边界里。

## 十二、本课最容易出现的误解

### 误解 1：`Config` 只是用户输入原样保存

不对。

它还会：

- 读取 `AutoConfig`
- 修正 `max_model_len`
- 承接后续补充信息

### 误解 2：`SamplingParams` 是全局设置

不对。

它是请求级的，后续多个请求甚至可以各自不同。

### 误解 3：`gpu_memory_utilization` 只是一个性能参数

不完全对。

它后面会影响：

- cache 容量
- 可调度空间

所以它会影响系统结构，不只是速度。

### 误解 4：`max_model_len` 就是你填多少用多少

不对。

模型真实能力会把它截断。

## 十三、建议的阅读动作

### 动作 1：把 `Config` 字段分组

请你自己把它分成：

- 路径与模型
- batch 边界
- 显存与执行策略
- 运行时推导字段
- cache 几何参数

### 动作 2：给每个字段补一句“它限制了什么”

不要只写定义，要写：

- 这个字段对系统行为有什么约束

### 动作 3：尝试用伪代码描述初始化顺序

比如：

1. 用户传入 `model` 和若干 kwargs
2. 构造 `Config`
3. `Config` 读取 `AutoConfig`
4. 后续运行时再补 `eos` 等信息

这会帮助你进入第 5 课。

## 十四、本课自测题

1. 为什么 `Config` 和 `SamplingParams` 不能合成一个对象？
2. `max_num_batched_tokens` 和 `max_model_len` 的关系为什么要被 assert？
3. 为什么说 `gpu_memory_utilization` 会影响系统结构？
4. `temperature > 1e-10` 这个限制说明了什么？
5. 哪些 `Config` 字段是后续运行时才真正填充出来的？

## 十五、本课作业

### 作业 1

用不超过 250 字解释：

> 为什么说 `Config` 描述的是引擎边界，而 `SamplingParams` 描述的是请求行为？

### 作业 2

给 `Config` 的每个字段各写一句用途说明。

### 作业 3

回答：

> 如果我要支持 greedy sampling，应该优先改 `Config` 还是 `SamplingParams`？为什么？

## 十六、本课最终收获

学完这一课，你应该已经具备了一种非常重要的阅读能力：

- 看到一个参数，不再只问“它是什么”
- 而是会问“它约束了什么，它属于哪一层”

这种能力会直接帮你在第 5 课读懂 `LLMEngine` 的初始化和主循环。

## 下一课预告

第 5 课会正式进入总入口 `LLM` 与 `LLMEngine`，重点回答：

- 为什么 `LLM` 几乎是空壳
- 真正的系统协调者是谁
- `generate()` 是如何驱动整个运行时循环的
