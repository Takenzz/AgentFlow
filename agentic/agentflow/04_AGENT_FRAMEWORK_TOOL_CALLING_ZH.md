# AgentFlow 整体框架、Agent 调度与工具调用详解

这份文档专门解释 AgentFlow 的“系统是怎么跑起来的”。如果面试官问你：一个题目进来以后，Agent 内部怎么调度？Planner 怎么选择工具？Executor 怎么把自然语言变成工具调用？Memory 怎么影响下一步？训练时又怎么把这些轨迹变成 RL 样本？你可以按这份文档回答。

这份文档假设你不是经验丰富的工程师，所以会尽量把每个概念拆开讲。

## 1. 先建立大图

AgentFlow 不是一个单独的大模型直接回答问题，而是一个多角色 Agent 系统。它把“解一道数学题”拆成几个角色协作：

```text
用户题目
  -> Planner：理解题目，决定下一步要做什么
  -> Executor：把 Planner 的决定翻译成工具命令
  -> Tool：执行局部推理或代码计算
  -> Memory：记录工具调用和结果
  -> Verifier：判断信息够不够，是否继续
  -> final_output：只汇总 Memory 中已有工具结果，给最终答案
  -> Rewarder：判断最终答案对不对
```

最核心的一句话：

> AgentFlow 训练的不是“直接答题模型”，而是“会调度工具、管理中间状态、决定何时停止的小 Planner”。

## 2. 代码入口在哪里

根据场景不同，入口不一样：

| 场景 | 入口文件 | 作用 |
|---|---|---|
| RL 训练 | `launch.sh` -> `agentflow_qwen25_7b_rl_v2.sh` -> `train.py` | 启动 slime/Ray 训练 |
| 训练 rollout | `rollout.py` 的 `generate()` | slime 每次采样时调用 |
| 训练 reward | `rollout.py` 的 `reward_func()` | 给采样结果打分 |
| 独立评测 | `eval_agentflow.sh` -> `eval_agentflow.py` | 不训练，只跑前向评测 |
| 单轮 baseline | `eval_baseline.sh` -> `eval_baseline.py` | 不使用 AgentFlow，直接 QA |

面试时可以这样说：

> 训练入口是 slime 的 `train.py`，但 AgentFlow 自己接入训练框架的关键点是 `rollout.py`。其中 `generate()` 负责生成一条多步 agent 轨迹，`reward_func()` 负责根据最终答案给 reward，`custom_convert.py` 再把多轮 Planner 输出转换成训练样本。

## 3. 一条样本的完整调度顺序

下面是 `Solver.solve()` 里实际发生的事情：

```text
输入 question

1. 创建 Memory
2. Planner.plan(question)
   -> 得到 analysis
   -> 这个 turn 参与 Planner 训练

3. 进入多步循环，最多 max_steps 次
   3.1 Planner.generate_next_step(question, memory, analysis)
       -> 得到 Context / Sub-Goal / Tool Name
       -> 这个 turn 参与 Planner 训练

   3.2 extract_context_subgoal_and_tool(...)
       -> 从 Planner 文本里解析出 context、sub_goal、tool_name

   3.3 Executor.generate_tool_command(...)
       -> 生成形如 execution = tool.execute(query="""...""") 的命令
       -> 不参与 Planner 训练

   3.4 Executor.execute_command(...)
       -> 加载对应 Tool
       -> 解析 query 参数
       -> 调用 tool.execute(query=...)
       -> 返回 execution_result

   3.5 Memory.add_action(...)
       -> 保存 tool_name、sub_goal、command、result

   3.6 Verifier.verificate_context(...)
       -> 根据 question、analysis、memory 判断 STOP 或 CONTINUE
       -> 不参与 Planner 训练

   3.7 如果 STOP，退出循环

4. Planner.generate_final_output(...)
   -> 使用 analysis + question + memory 生成最终答案
   -> 当前配置中 final_output 默认走 API，不参与 Planner 训练
   -> 它只能汇总 Memory 中已有工具结果；信息不足时应输出 `\boxed{INSUFFICIENT_TOOL_RESULTS}`

5. 返回 GenerationOutput
   -> 包含完整 response、final_output、Planner turns、token、logprob、loss_mask
```

## 4. 为什么需要 Solver

`Solver` 是整个 Agent 的调度器。你可以把它理解成一个流程控制器，不是模型本身。

它负责：

| 职责 | 具体做什么 |
|---|---|
| 初始化角色 | 创建 Planner、Executor、Verifier |
| 管理循环 | 最多跑 `max_steps` 个工具步骤 |
| 管理状态 | 创建和更新 Memory |
| 调用模型 | 按顺序调用 Planner、Executor、Verifier、final_output |
| 保存轨迹 | 可选写入 trajectory JSON |
| 组织训练数据 | 收集 Planner turns，生成 `GenerationOutput` |

面试表达：

> Solver 本身不学习，它像一个 Agent runtime。它规定什么时候调用 Planner，什么时候执行工具，什么时候让 Verifier 判断是否停止，最后把一条完整轨迹组织成训练框架能消费的数据。

## 5. Planner 到底输出什么

Planner 有三个核心能力：

### 5.1 初始分析 plan

输入：

```text
原始 question
可用工具列表
工具 metadata
```

输出：

```text
对题目的简短分析
需要哪些技能
可能用哪些工具
注意事项
```

这个阶段的作用不是最终答题，也不是写隐藏解法，而是建立通用的拆解和调度策略：先收集什么类型的局部结果、再做什么类型的局部计算或检查、什么证据足够停止。这样可以让大模型教师和小模型学生的差距体现在“拆得是否合理”，而不是谁在 analysis 里直接把题做完。

### 5.2 下一步决策 next_step

输入：

```text
question
query_analysis
available_tools
toolbox_metadata
previous_steps(memory)
```

要求输出格式：

```text
Justification: 为什么选这个工具
Context: 给工具的必要上下文
Sub-Goal: 这一步要完成的局部目标
Tool Name: 精确工具名
```

这一步是 AgentFlow 的核心。因为 Planner 的输出决定了：

| Planner 输出 | 后续影响 |
|---|---|
| `Context` | Executor 生成工具命令时能看到什么信息 |
| `Sub-Goal` | Tool 实际要做什么 |
| `Tool Name` | 加载哪个工具 |
| 输出格式 | 能不能被 parser 正确解析 |

当前 next_step prompt 明确要求 Sub-Goal 是一个局部操作：

| 目标 | 要求 |
|---|---|
| 推理类工具 | 只请求一个局部关系、转换、定理或一致性检查 |
| 计算类工具 | 只请求一个明确计算、枚举、化简或符号检查 |
| Context | 写清已知数据、已引入变量、可复用的工具结果和期望输出 |
| 禁止内容 | 不要求工具给最终答案、完整解法或整体策略 |

这个约束的目的，是让 Planner 必须学会把问题拆成可执行的小任务。工具可以完成局部动作，但不替 Planner 选择整题路线。

### 5.3 final_output

输入：

```text
analysis
question
memory actions
```

输出最终答案，要求以 `\boxed{}` 结尾。当前推荐配置中 final_output 默认用 API 支持模型生成，不训练本地 Planner 的 final_output 能力。这里有一个重要边界：final_output 只能根据 Memory 中已有工具结果做汇总，不能重新从原题开始完整解题；如果 Memory 不足，应输出 `\boxed{INSUFFICIENT_TOOL_RESULTS}`。

现在 final_output 的边界更严格：

| 情况 | 行为 |
|---|---|
| Memory 已有可靠 final candidate | 汇总并输出该候选 |
| Memory 只有中间量，还缺最后计算 | 输出 `INSUFFICIENT_TOOL_RESULTS` |
| Memory 中工具结果互相矛盾 | 输出 `INSUFFICIENT_TOOL_RESULTS` |
| 工具结果明显有误但没有新工具修正 | 不能靠 final_output 自己纠错 |

这样做是为了避免 API final_output 把 Planner 的错误路径救回来，导致 reward 归因不清。

## 6. 工具发现是怎么做的

Planner 初始化时会扫描 `tools_dir`：

```text
agentic/agentflow/tools/
  base.py
  base_generator/tool.py
  python_coder/tool.py
```

每个工具文件提供：

| 字段 | 作用 |
|---|---|
| `TOOL_NAME` | 对外暴露的工具名 |
| `TOOL_DESCRIPTION` | 给 Planner 的工具说明 |
| Tool class | 真正执行 `execute()` 的类 |

工具发现流程：

```text
Planner._discover_tools(tools_dir)
  -> 遍历所有 tool.py
  -> 加载 TOOL_NAME
  -> 加载 TOOL_DESCRIPTION
  -> 生成 available_tools
  -> 生成 toolbox_metadata
```

这些信息会写入 Planner prompt，所以 Planner 才知道有哪些工具可选。

## 7. 当前有哪些工具

当前主要有两个工具：

| 工具 | 代码 | 用途 |
|---|---|---|
| `Local_Math_Deduction_Tool` | `tools/base_generator/tool.py` | 做一个局部数学关系、定理、推理检查 |
| `Python_Code_Generator_Tool` | `tools/python_coder/tool.py` | 生成并执行 Python 代码，做明确计算或符号检查 |

### 7.1 Local_Math_Deduction_Tool

这个工具适合：

| 适合 | 不适合 |
|---|---|
| 推导一个局部公式 | 直接解完整题 |
| 检查一个小定理 | 给最终答案 |
| 说明一个中间关系 | 写长篇完整证明 |

它的设计刻意限制工具不要直接把整题做完，这样 Planner 仍然要负责规划，而不是把所有任务甩给工具。

实现上增加了 broad request 拒绝：

```text
完整解题 / 最终答案 / 过宽目标
  -> NEEDS_SMALLER_SUBGOAL
```

这不是为了让工具变弱，而是为了让 Planner 的子目标质量变成可训练信号。好的 Planner 会把请求缩小到一个局部关系或检查；差的 Planner 会得到拒绝结果，并需要在下一步恢复。

### 7.2 Python_Code_Generator_Tool

这个工具适合：

| 适合 | 不适合 |
|---|---|
| 数值计算 | 没有明确目标的完整解题 |
| 枚举验证 | 需要 GUI 或网络的任务 |
| 符号检查 | 超长、超慢程序 |

这个工具现在要求 query 里必须有明确的局部计算契约：

| 必须清楚 | 说明 |
|---|---|
| 输入 | 已知数字、表达式、变量范围或有限域 |
| 约束 | 要过滤、检查或满足的条件 |
| 输出 | 打印哪个中间值、计数、列表或布尔检查 |

如果请求是“求最终答案”“解完整问题”或者缺少具体输入/约束/输出，工具会返回：

```text
NEEDS_NUMERIC_SUBGOAL
```

这样 Python tool 更像执行仪器，而不是会自动设计算法的专家。Planner 必须负责把计算任务描述到可执行的粒度。

流程是：

```text
query 自然语言
  -> LLM 生成 Python code
  -> 提取代码块
  -> 清理危险调用
  -> 子进程执行
  -> 返回 stdout 或 error
```

## 8. 从 Planner 文本到工具调用

这是面试很容易问的地方：Planner 只是输出自然语言，怎么真的调用工具？

完整流程：

```text
Planner response
  -> formatters.extract_context_subgoal_and_tool()
     解析 Context / Sub-Goal / Tool Name
  -> Executor.generate_tool_command()
     生成 Python command
  -> Executor._extract_command()
     从 LLM 输出中提取代码块
  -> Executor._parse_command_kwargs()
     拿到 tool.execute(query=...) 里的 query
  -> Executor._load_tool()
     根据 tool_name 加载工具类
  -> tool.execute(query=query)
     真正执行工具
```

关键防线：

| 防线 | 作用 |
|---|---|
| 固定 Planner 输出格式 | 让 parser 能抽取字段 |
| `extract_context_subgoal_and_tool` | 容忍 Markdown、编号、特殊 token |
| Executor command prompt | 强制只调用 `tool.execute(query=...)` |
| Executor format examples | 只锚定命令格式，不提供具体解题模板 |
| `_parse_command_kwargs` | 用 fake tool 或 regex 抽取参数 |
| `_resolve_tool_mapping` | 默认严格校验工具名；只有设置 `AGENTFLOW_ALLOW_TOOL_ALIASES=true` 时才兼容旧别名 |
| `execute_command` 捕获异常 | 工具失败不让整条轨迹崩溃 |

## 9. 为什么 Executor 要单独存在

如果没有 Executor，Planner 就必须直接输出可执行 Python 命令。这会让 Planner 学两个东西：

1. 如何规划下一步。
2. 如何写符合工具接口的代码。

这会增加训练难度。拆出 Executor 后：

| Planner | Executor |
|---|---|
| 负责语义规划 | 负责格式转换 |
| 输出自然语言 Context/Sub-Goal/Tool | 输出 `tool.execute(query=...)` |
| 参与 RL 训练 | 不参与训练 |
| 学会决策 | 保证工具接口稳定 |

面试表达：

> 我把策略决策和工具调用格式解耦。Planner 只需要学“下一步做什么”，Executor 负责把这个意图翻译成工具接口，从而降低 Planner 的学习难度，也让工具调用更稳定。

Executor prompt 里仍然保留格式示例，因为解析器依赖稳定的命令形态：

````text
Generated Command:
```python
execution = tool.execute(query="""...""")
```
````

但示例内容是通用占位，只说明格式，不包含具体题型或解题路径。这样能兼顾解析稳定性和泛化性。

## 10. Memory 怎么影响下一步

每次工具执行后，Solver 会调用：

```text
memory.add_action(step_count, tool_name, sub_goal, command, execution_result)
```

Memory 保存成类似：

```json
{
  "Action Step 0": {
    "tool_name": "Python_Code_Generator_Tool",
    "sub_goal": "Compute the candidate values",
    "command": "execution = tool.execute(query=\"\"\"...\"\"\")",
    "result": "..."
  }
}
```

下一次 Planner.generate_next_step() 会看到：

```text
Previous Steps: memory.get_actions()
```

所以 Memory 的作用是：

| 作用 | 解释 |
|---|---|
| 防止重复做同一步 | Planner 能看到历史工具调用 |
| 提供工具结果 | Planner 可以基于结果继续推理 |
| 支持 Verifier 判断 | Verifier 看 memory 是否足够 |
| 支持 final_output 汇总 | 最终答案根据 memory 写出 |

Memory 也会截断过长结果，避免上下文爆炸。

## 11. Verifier 如何调度 Agent 循环

Verifier 每一步看到：

```text
question
available_tools
toolbox_metadata
initial analysis
memory
```

它只做一个决策：

```text
Conclusion: STOP
或
Conclusion: CONTINUE
```

如果 STOP：

```text
跳出工具循环 -> final_output
```

如果 CONTINUE：

```text
回到 Planner.generate_next_step()
```

Verifier 的意义：

| 没有 Verifier | 有 Verifier |
|---|---|
| 固定跑满 max_steps | 可以提前停止 |
| 容易浪费 API 和上下文 | 成本更可控 |
| Planner 不知道何时结束 | 有外部停止判断 |
| 错误步骤可能累积 | 可在信息不足时继续补工具 |

当前 STOP 标准也被收紧：只有 Memory 已经包含可靠 final candidate，或者包含所有必要中间值并且已经有明确记录的最终量计算，才允许 STOP。遇到工具错误、`NEEDS_SMALLER_SUBGOAL`、`NEEDS_NUMERIC_SUBGOAL`、不完整结果或同一目标的矛盾结果时，Verifier 应继续让 Planner 补一个更小的局部子目标。

风险：

| 风险 | 表现 | 处理 |
|---|---|---|
| 过早 STOP | final_output 信息不足 | 调 Verifier prompt，加入 OPD 停止样本 |
| 一直 CONTINUE | 跑满 max_steps | 限制 max_steps，加入停止训练样本 |
| 判断不稳定 | 同类题忽停忽继续 | 降低温度或换更稳 judge |

## 12. final_output 为什么单独做

Planner 的 plan/next_step 更像“过程决策”，final_output 是“答案表达”。这两个能力不同。

当前设计中 final_output 默认由 API 生成，但它的角色是“汇总器”，不是“重新解题器”。原因是：

| 原因 | 说明 |
|---|---|
| 稳定答案格式 | 更容易输出 `\boxed{}` |
| 降低训练目标复杂度 | Planner 专注调度 |
| 避免无 logprob 问题 | API 输出不参与 loss |
| 方便比较 Planner 能力 | 支持角色固定，只换 Planner |

为了避免 API final_output 绕过 Planner 和工具直接解题，prompt 中要求它只能使用 Actions Taken 里的信息；如果工具结果不足，需要明确输出不足，而不是自己补完整推理。

这次同步后，final_output 还不能做“工具结果修复”：如果 Memory 内部矛盾或缺少最后一步计算，它必须输出不足。这样最终 reward 更依赖 Planner 是否真的把工具链走完整。

如果未来想训练 final_output，就需要让 final_output 也由本地 Planner 生成，并正确记录 token/logprob/loss_mask。

## 13. 训练时哪些 token 参与 loss

参与 loss：

```text
Planner.plan() 的 response token
Planner.generate_next_step() 的 response token
```

不参与 loss：

```text
Executor 输出
Tool 输出
Verifier 输出
Memory 文本
final_output API 输出
Rewarder 输出
```

原因：

1. RL 更新的是本地 Planner。
2. Executor/Verifier/Rewarder 多数走 API，没有本地可训练 logprob。
3. 工具结果是环境反馈，不是模型输出。
4. final_output 当前用于形成 reward，但不是被训练目标。

## 14. 训练轨迹如何变成 GRPO 样本

一条 trajectory 可能长这样：

```text
plan turn
next_step turn 0
tool result 0
verifier 0
next_step turn 1
tool result 1
verifier 1
final_output
reward
```

`Solver` 会记录 Planner turns：

```text
turns = [
  plan_turn,
  next_step_turn_0,
  next_step_turn_1
]
```

`custom_convert.py` 会：

```text
1. 读取每条 sample 的最终 reward
2. 对同 prompt 的多条采样做 reward normalization
3. 把每条 trajectory 的 turns 展开
4. 给同一 trajectory 的每个 turn 分配同一个 normalized reward / turn_count
5. 输出 tokens、response_lengths、loss_masks、rewards、rollout_log_probs
```

面试表达：

> 我们的 reward 是整条轨迹的最终 reward，但训练样本是每个 Planner turn。custom_convert 负责把 trajectory-level reward 分摊到各个 Planner 决策上，让 GRPO 能优化多步 agent 的每个规划动作。

## 15. 评测时和训练时有什么区别

| 项 | 训练 | 评测 |
|---|---|---|
| 入口 | `rollout.generate()` | `eval_agentflow.py` |
| 是否更新参数 | 是 | 否 |
| 是否需要 logprob | 是 | 否 |
| Planner | 本地 SGLang | 本地 SGLang 或 API |
| 支持角色 | API | API |
| 输出 | sample + reward + train metadata | JSON 结果和轨迹 |
| 目的 | 更新 Planner | 计算 accuracy 和分析 case |

评测时可以使用 API Planner，因为它只做前向推理，不需要训练 logprob。

训练时不能直接用 API Planner，因为闭源 API 的 token/logprob 和本地模型参数不可回传。

## 16. 面试官可能追问的调度问题

### Q1：Planner 输出错了工具名怎么办？

默认会报工具加载错误，并把错误写回 Memory，不会静默路由到通用工具。只有评测旧 checkpoint 时，才建议显式设置 `AGENTFLOW_ALLOW_TOOL_ALIASES=true` 兼容旧工具名。根本解决还是通过 SFT/OPD 让 Planner 输出 `Available Tools` 里的精确工具名。

### Q2：工具执行失败会不会导致整条训练崩？

不会。`execute_command()` 捕获工具加载、命令解析、工具执行异常，并把错误作为字符串返回给 Memory。后续 Planner/Verifier 仍然可以基于这个错误继续或停止。

### Q3：Memory 里有错误结果怎么办？

这正是 Agent 训练难点。Planner 下一步会看到错误结果，可能需要学会恢复。OPD 很适合处理这种情况：让教师在学生真实错误 memory 上给出更好的下一步。

### Q4：为什么不让工具直接返回最终答案？

如果工具直接解完整题，Planner 学不到规划能力，系统退化成“把题转发给工具”。所以 `Local_Math_Deduction_Tool` 明确限制只做局部推理，`Python_Code_Generator_Tool` 也只接受明确输入、约束和输出的局部计算。过宽请求会以 `NEEDS_SMALLER_SUBGOAL` 或 `NEEDS_NUMERIC_SUBGOAL` 暴露出来，给 Planner 训练提供可观察的失败信号。

### Q5：Verifier 是不是也应该训练？

可以作为未来扩展，但当前不训练。当前项目为了资源和目标清晰，把 Verifier 当固定环境。等 Planner 稳定后，可以考虑收集 STOP/CONTINUE 数据单独训练 Verifier。

### Q6：为什么要把 Executor 和 Tool 分开？

Executor 是“翻译器”，Tool 是“执行器”。Executor 负责把 Planner 意图转换成统一接口，Tool 负责真正计算或推理。这样工具接口可以保持一致，Planner 不需要学每个工具的调用细节。

### Q7：如果 final_output 是 API 生成，那提升是不是来自 API？

要做对照实验。固定 final_output、Executor、Verifier、Rewarder，只替换 Planner checkpoint。如果 RL 后 Planner 的指标提高，就说明更好的 plan/next_step 让固定支持角色拿到了更好的中间信息。

## 17. 你可以这样画流程图

```text
                         +-------------------+
                         | question + label  |
                         +---------+---------+
                                   |
                                   v
                         +-------------------+
                         | Planner.plan      |
                         | output analysis   |
                         +---------+---------+
                                   |
                                   v
             +---------------------+----------------------+
             |                                            |
             v                                            |
   +----------------------+                               |
   | Planner.next_step    |<------------------------------+
   | Context/SubGoal/Tool |
   +----------+-----------+
              |
              v
   +----------------------+
   | parse planner output |
   +----------+-----------+
              |
              v
   +----------------------+
   | Executor command     |
   | tool.execute(query)  |
   +----------+-----------+
              |
              v
   +----------------------+
   | Tool execution       |
   +----------+-----------+
              |
              v
   +----------------------+
   | Memory.add_action    |
   +----------+-----------+
              |
              v
   +----------------------+
   | Verifier STOP?       |
   +----+-------------+---+
        |             |
      no|             |yes
        +-------------+ 
                      |
                      v
            +-------------------+
            | final_output      |
            +---------+---------+
                      |
                      v
            +-------------------+
            | Rewarder score    |
            +---------+---------+
                      |
                      v
            +-------------------+
            | GRPO update       |
            +-------------------+
```

## 18. 初学者理解版类比

可以把 AgentFlow 想成一个做题小组：

| 角色 | 类比 |
|---|---|
| Planner | 组长，决定下一步找谁做什么 |
| Executor | 助教，把组长的话改写成标准任务单 |
| Tool | 专门干活的人，比如计算员或定理查询员 |
| Memory | 会议记录，记录每一步做了什么 |
| Verifier | 检查员，判断材料够不够交卷 |
| final_output | 写答案的人，只把会议记录里已有材料整理成最终答案 |
| Rewarder | 阅卷老师，判断答案对不对 |

训练时真正被培养的是“组长”。其他人暂时固定，保证组长能在稳定环境里学会调度。

## 19. 面试复述模板

你可以背这个版本：

> 一道题进来后，Solver 先创建 Memory，然后调用 Planner 做初始 analysis。接着进入多步循环，Planner 根据原题、analysis 和 memory 生成下一步决策，包括 Context、Sub-Goal 和 Tool Name。系统用 parser 抽取这些字段，Executor 再把它翻译成统一的 `tool.execute(query=...)` 命令。Tool 执行后结果写入 Memory，Verifier 根据当前 Memory 判断 STOP 还是 CONTINUE。如果继续，就回到 Planner 生成下一步；如果停止，就由 final_output 根据 Memory 中已有工具结果生成最终答案；如果信息不足，应输出不足而不是重新解题。Rewarder 比较最终答案和 label 得到 reward。训练时只把 Planner 的 plan 和 next_step token 纳入 loss，Executor、Tool、Verifier、final_output 和 Rewarder 都作为环境或固定支持模块。

这个回答能覆盖：调度、工具调用、状态管理、停止判断、reward、训练目标。
