# 第 3 课：从示例程序认识外部 API

## 本课在整套课程中的位置

前两课解决了两个基础问题：

- 这个项目是什么
- 这个项目依赖什么环境

从这一课开始，我们正式进入“怎么使用它，以及一个请求是怎样进入系统的”。

第 3 课不直接去看调度器和 KV cache，而是先站在使用者视角，读最小示例程序。这样做的好处是：

- 你先知道用户到底怎么调用这个引擎
- 你先知道哪些对象是系统公开暴露的接口
- 你先知道哪一层是“外部 API”，哪一层是“内部运行时”

这一课本质上是在回答：

> 如果我是这个项目的使用者，我是怎么把一段文本变成一次推理请求的？

## 本课目标

学完这一课后，你应该能够：

1. 说清楚 `example.py` 从上到下在做什么
2. 区分 `LLM`、`SamplingParams`、tokenizer 各自的职责
3. 理解 prompt 为什么在示例里先经过 chat template
4. 说清楚 `generate()` 的输入和输出长什么样

## 本课涉及文件

- `example.py`
- `nanovllm/llm.py`
- `nanovllm/engine/llm_engine.py`
- `nanovllm/sampling_params.py`

## 先给出结论

这一课最重要的结论有 4 个：

1. 对外最核心的 API 只有两个：`LLM` 和 `SamplingParams`
2. tokenizer 不属于 nano-vllm 自己的 API 核心，而是借助 `transformers` 完成文本模板化和编码
3. `generate()` 支持批量请求，这是后面调度器存在的前提
4. 这个示例故意把内部复杂度藏起来了，所以它看起来很简单，但背后并不简单

## 一、先完整读一遍 `example.py`

这个示例程序的结构非常短，但信息量很高：

1. 指定模型路径
2. 构造 tokenizer
3. 构造 `LLM`
4. 构造 `SamplingParams`
5. 准备 prompts
6. 通过 chat template 生成最终输入文本
7. 调用 `llm.generate(...)`
8. 打印输出

你第一次读的时候，不要急着问内部怎么实现，先回答：

- 外部调用者手里到底要准备哪些东西
- 哪些对象是用户主动创建的
- 哪些东西是引擎自己在内部接管的

## 二、模型路径：系统从哪里知道要加载什么

示例一开始就写了：

- `path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")`

这说明一件重要的事：

这个项目默认期望你已经把模型权重下载到本地目录，而不是在运行时临时联网下载。

所以在使用视角下，`model path` 是第一层输入。

它后续会影响：

- `AutoTokenizer.from_pretrained(path)`
- `AutoConfig.from_pretrained(path)`
- 权重加载逻辑

换句话说，模型路径不是一个普通字符串，而是整个运行时初始化的根起点。

## 三、tokenizer 的角色：它不负责推理，但负责把输入变对

示例里先构造了：

- `tokenizer = AutoTokenizer.from_pretrained(path)`

然后又调用了：

- `tokenizer.apply_chat_template(...)`

这一段很关键，因为它会纠正一个常见误解：

> LLM 引擎只负责“高效生成”，不负责替你决定 prompt 应该长什么样。

也就是说：

- nano-vllm 负责高效推理
- tokenizer 负责把人类可读输入整理成模型习惯的格式

这里的 chat template 尤其重要，因为很多 chat 模型并不是直接接收自然语言句子，而是接收一段带有角色标记、起始标记、生成提示的模板化文本。

因此示例里这一步不是“装饰”，而是：

- 让 prompt 更符合模型预期
- 保证输入格式和训练时的对话模板一致

## 四、`LLM` 到底是什么

示例里最核心的一行是：

- `llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)`

这里你需要先形成两个初步认识。

### 1. `LLM` 是用户真正接触的主入口

对外使用者并不直接创建：

- `Scheduler`
- `ModelRunner`
- `Sequence`
- `BlockManager`

而是只创建一个 `LLM`。

也就是说，`LLM` 是系统的门面对象。

### 2. 这两个参数非常有教学意义

- `enforce_eager=True`
- `tensor_parallel_size=1`

这两个设置本质上是在故意选择一条更容易学习的路径：

- 先不用 CUDA Graph
- 先不用多卡并行

所以你应该把这个示例理解成：

> 它不是在展示“最强配置”，而是在展示“最适合第一次理解的配置”。

## 五、`SamplingParams`：为什么采样参数是单独对象

示例里创建了：

- `sampling_params = SamplingParams(temperature=0.6, max_tokens=256)`

这一点非常重要，因为它说明系统把两类信息分开了：

- 模型与运行时配置
- 单次生成请求的采样配置

这两个东西不能混为一谈。

比如：

- 模型路径、并行度、最大上下文长度，更像“引擎级设置”
- `temperature`、`max_tokens`、`ignore_eos`，更像“请求级设置”

所以 `SamplingParams` 的存在，本质上是在表达：

> 采样行为属于请求，而不是属于整个引擎实例。

这个分层在后面你会反复看到。

## 六、为什么示例要用两个 prompt

示例没有只写一个 prompt，而是写了两个：

- `"introduce yourself"`
- `"list all prime numbers within 100"`

这不是随手写的，它有很强的教学意义。

它在暗示你：

- `generate()` 本来就支持批量输入
- 这个项目天然就是朝“并发处理多个请求”设计的

也就是说，示例虽然看起来轻量，但它已经把系统最关键的思想暴露出来了：

- 推理不是单条请求逐个做完
- 推理引擎天生要考虑 batch

这也是你后面学习 `Scheduler` 时最重要的前提。

## 七、chat template 之后，prompt 还是字符串吗

是的。

示例里：

- `apply_chat_template(..., tokenize=False, add_generation_prompt=True)`

这里 `tokenize=False` 的意思是：

- 先生成格式化文本
- 不在这一步直接转成 token ids

所以传给 `llm.generate()` 的，仍然是一组字符串。

之后系统内部才会在适当时机对字符串做 tokenize。

这个细节很重要，因为它告诉你：

- 外部 API 支持字符串输入
- 但内部运行时最终仍然要变成 token ids

这两层是分开的。

## 八、`generate()` 的输入形式是什么

从示例看，`generate()` 的输入包括两部分：

### 1. prompts

这里是一个列表：

- `list[str]`

但内部实现还支持：

- `list[list[int]]`

也就是你既可以传字符串，也可以传已经分好词的 token ids。

### 2. sampling_params

示例传的是一个单独的 `SamplingParams` 对象。

这意味着：

- 所有请求共享同一份采样设置

但在内部实现里，它还支持：

- 每个 prompt 对应一份不同的 `SamplingParams`

这会在后面看到。

所以第 3 课你要先建立一个 API 认知：

> 这个项目的 `generate()` 已经具备面向批量请求和不同调用方式的基本弹性。

## 九、`generate()` 的输出是什么

示例里最后打印的是：

- `output['text']`

这说明返回值不是裸字符串列表，而是结构化结果。

在内部实现中，每个输出大致包含：

- `text`
- `token_ids`

也就是说，系统默认保留：

- 人类可读文本
- 机器可读 token 序列

这个设计很合理，因为学习和调试推理引擎时，保留 token ids 非常有用。

## 十、这个示例故意隐藏了哪些复杂度

这部分非常重要。

你不能因为示例只有十几行，就误以为它背后也很简单。

这个示例至少隐藏了下面这些内部复杂度：

- tokenizer 最终何时被调用
- prompt 如何变成 `Sequence`
- 多个请求如何一起进入调度器
- 是 prefill 还是 decode，由谁决定
- token 是怎么一步步追加到结果里的

所以这一课的关键能力，不是“会照抄示例”，而是：

> 能透过这个简单示例，看到它后面埋着一整条运行时主线。

## 十一、使用视角下的 4 个对象

建议你现在就把外部使用中的 4 个对象分清楚。

### 1. 模型路径

作用：

- 告诉系统去哪里找模型配置、tokenizer 和权重

### 2. tokenizer

作用：

- 负责文本预处理
- 负责 chat template
- 负责后续文本解码

### 3. `LLM`

作用：

- 作为推理引擎门面
- 管理内部运行时组件

### 4. `SamplingParams`

作用：

- 控制一次生成请求的采样行为

你如果能把这 4 个对象的职责说清楚，这一课就算真正掌握了。

## 十二、本课最容易出现的误解

### 误解 1：tokenizer 是 nano-vllm 的一部分

不准确。

这里 tokenizer 来自 `transformers` 生态，nano-vllm 是在使用它，而不是自己重新实现它。

### 误解 2：`generate()` 只适合单条请求

不对。

从示例就能看出，它天然支持批量输入。

### 误解 3：chat template 是可有可无的字符串拼接

不对。

对于 chat 模型来说，模板直接影响输入格式是否符合模型预期。

### 误解 4：外部 API 简单，就代表内部实现简单

不对。

恰恰相反，好的推理引擎通常是：

- 对外接口尽量简单
- 对内运行时复杂度集中封装

## 十三、建议的阅读动作

这一课建议你做 3 件事。

### 动作 1：手工标注示例程序

给 `example.py` 的每一行都标注它属于哪一层：

- 输入准备
- tokenizer
- 引擎初始化
- 采样设置
- 发起生成
- 输出展示

### 动作 2：把示例改成 token ids 输入

尝试思考：

- 如果不传字符串，而是直接传 token ids，会绕过哪一层？

这会帮助你区分“文本接口”和“底层运行时接口”。

### 动作 3：把两个 prompt 换成长度差异很大的请求

然后带着问题进入后续课程：

- 为什么不同长度请求还能放在同一个 `generate()` 里？

这个问题会直接把你带向 `Scheduler`。

## 十四、本课自测题

1. `example.py` 里谁负责决定 prompt 的文本格式？
2. `LLM` 和 tokenizer 的职责差别是什么？
3. 为什么 `SamplingParams` 要独立成对象？
4. `generate()` 为什么能接受多个 prompt？
5. 为什么说这个示例已经暗示了后面会有调度器？

## 十五、本课作业

### 作业 1

用不超过 200 字解释：

> 为什么 `example.py` 虽然很短，但已经暴露了这个项目是一个“批量推理引擎”？

### 作业 2

画一张小图，说明这 4 个对象的关系：

- model path
- tokenizer
- `LLM`
- `SamplingParams`

### 作业 3

回答下面这句话对不对，并解释原因：

> “只要我会用 `example.py`，我就已经理解了 nano-vllm 的核心。”

## 十六、本课最终收获

学完这一课，你最重要的收获不是“我会跑示例”，而是：

- 你已经知道系统对外暴露了什么
- 你知道外部 API 和内部运行时之间存在一层清晰边界
- 你知道下一步应该进入配置对象和总入口，而不是直接跳到底层算子

## 下一课预告

第 4 课会进入 `Config` 和 `SamplingParams`，重点解决：

- 哪些参数属于引擎级配置
- 哪些参数属于请求级配置
- 这些配置在系统里是怎样约束运行边界的
