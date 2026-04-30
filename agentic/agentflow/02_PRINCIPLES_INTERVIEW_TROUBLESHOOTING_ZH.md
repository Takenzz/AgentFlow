# AgentFlow 原理、整体链路、面试问题与训练问题排查

这份文档是三份文档里最重要的一份。它不只是记录命令，而是解释 AgentFlow 为什么这样设计、训练时每个模块发生了什么、面试中可能怎么问，以及 SFT、OPD、RL 各阶段容易出什么问题、怎么定位和解决。

## 1. 一句话概括项目

AgentFlow 是一个基于 slime 的多步 agent 训练项目：用本地小模型作为 Planner，让它在数学推理任务中学习如何分析问题、选择工具、执行子目标、判断是否继续，并最终生成答案；训练阶段用 GRPO 根据最终任务 reward 更新 Planner，其他辅助角色默认由 OpenAI 兼容 API 提供。

更适合面试的表述：

> 我没有把大模型整体蒸馏成小模型，而是把 agent 系统拆成可训练的 Planner 和固定的支持角色。小 Planner 负责决策和工具规划，Executor、Verifier、Rewarder 由更强的 API 模型承担。训练时只优化 Planner 的多轮决策 token，reward 来自最终答案是否正确。这样在有限 GPU 下，也能训练一个具备多步规划能力的小模型。

## 2. 模块职责

| 模块 | 代码位置 | 职责 | 训练时是否更新 |
|---|---|---|---:|
| Planner | `core/planner.py` | 只做任务拆解、选择下一步工具、给出 sub-goal，并最终汇总 Memory | 是 |
| Solver | `core/solver.py` | 串起 plan、next_step、execute、verify、final_output | 否，本身是流程控制 |
| Executor | `core/executor.py` | 把 Planner 的工具选择转成 `tool.execute(query=...)` | 否 |
| Verifier | `core/verifier.py` | 判断已有 memory 是否足够，输出 STOP/CONTINUE | 否 |
| Memory | `core/memory.py` | 保存工具调用、子目标、执行结果 | 否 |
| Tools | `tools/*/tool.py` | 完成局部推理、局部计算或工具执行 | 否 |
| Rewarder | `core/rewarder.py` | 判断最终答案和 label 是否等价 | 否 |
| rollout | `rollout.py` | slime 训练时的自定义生成和 reward 函数 | 否 |
| custom_convert | `custom_convert.py` | 把多轮 Planner turn 展开成训练样本 | 否 |

## 3. 推理链路详解

一条样本的完整前向过程如下：

```text
question
  -> Planner.plan(question)
       输出初始 analysis
  -> for step in max_steps:
       Planner.generate_next_step(question, analysis, memory)
         输出 Justification / Context / Sub-Goal / Tool Name
       Executor.generate_tool_command(...)
         输出 Python 形式的 tool.execute(query=...)
       Executor.execute_command(...)
         调用 Local_Math_Deduction_Tool 或 python_coder
       Memory.add_action(...)
         保存 tool_name / sub_goal / command / result
       Verifier.verificate_context(...)
         输出 Conclusion: STOP 或 CONTINUE
       如果 STOP 则跳出
  -> Planner.generate_final_output(...)
       只根据 memory 中已有工具结果生成最终答案
  -> Rewarder.compute_reward(...)
       比较 final_output 和 label，返回 0/1 reward
```

关键点：

1. Planner 是被训练对象，决定 agent 下一步做什么。
2. Planner 不应该直接解题，它只输出当前要让哪个工具完成哪个局部目标。
3. Executor 不直接决定解题策略，它把 Planner 的自然语言决策变成工具调用。
4. Tool 执行结果写入 Memory，下一步 Planner 会看到历史 actions。
5. 工具不会替 Planner 兜底完成整题：局部推理工具只给局部关系，代码工具只做明确计算。
6. Verifier 控制循环是否继续，避免无意义多步调用。
7. final_output 默认由 API base model 生成，不参与 Planner loss，但只能汇总工具结果，信息不足时应输出不足。
8. reward 只看最终答案是否正确，不逐步奖励每个工具调用。
9. 工具名默认严格匹配 `Available Tools`，不会把未知工具名自动路由到 base_generator；旧 checkpoint 兼容需要显式设置 `AGENTFLOW_ALLOW_TOOL_ALIASES=true`。

## 4. 为什么只训练 Planner

因为资源有限，而且 agent 能力的核心瓶颈通常是规划决策，不是所有角色都必须本地训练。

训练全部模块会遇到：

| 问题 | 影响 |
|---|---|
| 显存占用高 | 单机/少卡很难同时训练和部署多个模型 |
| credit assignment 更复杂 | Executor、Verifier、Planner 的错误互相耦合 |
| logprob 难对齐 | API 或异构模型不返回可训练 token logprob |
| 工具输出不可微 | 只能通过最终 reward 间接优化 |
| 工程复杂度高 | 多服务同步、checkpoint、版本控制都更难 |

只训练 Planner 的好处：

| 好处 | 解释 |
|---|---|
| 训练目标清晰 | 学的是“何时调用什么工具、带什么上下文、何时停止” |
| 显存可控 | 本地 GPU 留给小 Planner |
| 支持大模型辅助 | Executor/Verifier/Rewarder 可用更强 API |
| 容易做对照 | API Planner 是上限，本地 Planner 是被优化对象 |
| 面试讲法清楚 | 符合系统拆分、资源约束和 RL 优化逻辑 |

## 5. Planner 的训练样本是什么

Planner 在一条 trajectory 中产生多个可训练 turn：

1. `plan` turn：对原题做初始分析。
2. `next_step` turn 1：基于空 memory 选择第一步工具。
3. `next_step` turn 2：基于已有工具结果选择下一步。
4. 后续 `next_step` turns：直到 STOP 或达到 `max_steps`。

`Solver.solve()` 会把这些 turn 存进 `GenerationOutput.turns`。每个 turn 包含：

| 字段 | 说明 |
|---|---|
| `tokens` | prompt token + response token |
| `response_length` | Planner response token 长度 |
| `loss_mask` | 哪些 token 参与 loss |
| `rollout_log_probs` | rollout 时旧策略 logprob |

Executor、Verifier、工具结果和 final_output 虽然影响最终 reward，但它们不放进 Planner 的 loss。

## 6. custom_convert 为什么重要

多轮 agent trajectory 不是普通单轮 completion。如果直接把所有文本拼成一个长 response，会有两个问题：

1. 工具结果、Verifier 输出、final_output 不是 Planner 生成的，不应该让 Planner 学。
2. 多个 Planner turn 之间有不同 prompt，如果简单拼接，loss mask 和 logprob 很容易错位。

`custom_convert.py` 的做法：

1. 先对一组 samples 做 GRPO reward normalization。
2. 如果 sample 里有 `train_metadata["turns"]`，就把每个 turn 展开成独立训练样本。
3. 同一条 trajectory 的 normalized reward 会分配给它的所有 turns。
4. 每个 turn 的 loss mask 只覆盖 Planner 生成 token。
5. 展开后样本数可能不再是 global batch size 的整数倍，所以会 trim 到合适长度。

这对应一个很重要的思想：最终 reward 是 trajectory-level 的，但优化对象是每个 Planner 决策 turn。

## 7. GRPO 在这里做什么

GRPO 的核心是：对同一个 prompt 采样多条输出，用组内 reward 差异估计 advantage，然后提高高 reward 输出的概率、降低低 reward 输出的概率。

在 AgentFlow 中：

```text
同一道数学题
  -> Planner 采样 N 条不同 agent trajectory
  -> 每条 trajectory 得到最终 reward
  -> 组内归一化得到 advantage
  -> 每条 trajectory 的 Planner turns 共享该 advantage
  -> 更新 Planner
```

如果一条 trajectory 最终答对，它内部的 plan 和 next_step 会得到正向信号；如果答错，就得到负向信号。虽然 reward 不是步骤级的，但多采样和组内比较能让模型逐渐偏向更有效的规划路径。

## 8. reward 是怎么来的

Rewarder 会读取：

| 输入 | 来源 |
|---|---|
| `question` | 原始题目 |
| `model_response` | final_output |
| `groundtruth` | label |

判断规则：

1. 优先抽取 `\boxed{...}`。
2. 如果 pred 和 label 字符串精确匹配，直接给 1。
3. 否则调用 LLM judge，要求输出 `VERDICT: True` 或 `VERDICT: False`。
4. 解析失败默认 0。

风险：

| 风险 | 解决 |
|---|---|
| LLM judge 过宽 | prompt 中要求严格判断；必要时换规则判分 |
| `\boxed{}` 格式缺失 | 在 final_output prompt 和 SFT 数据中强化格式 |
| 等价答案未识别 | 加 normalize、sympy 或数学规则判分 |
| API 抖动 | 使用 timeout、retry、缓存和小样本复查 |

## 9. SFT、OPD、RL 的关系

推荐顺序：

```text
原始 Planner baseline
  -> 可选 SFT 热启动
  -> 可选 OPD on-policy 教师纠偏
  -> GRPO RL
  -> checkpoint 评测和轨迹分析
```

### 9.1 SFT 是什么

SFT 让小 Planner 先学会基本格式和协议，例如：

| 需要学会的内容 | 例子 |
|---|---|
| 初始分析格式 | 能简短识别目标和可拆分的中间量，不直接解题 |
| next_step 格式 | 输出 Context / Sub-Goal / Tool Name |
| 工具选择 | 局部关系用 Local_Math_Deduction_Tool，数值计算用 python_coder |
| 停止意识 | 信息足够时不要继续调用工具 |
| 答案格式 | 最终答案需要 `\boxed{}` |

SFT 的目标不是直接把题答对，而是减少 RL 初期因为格式错误导致的无效探索。

### 9.2 OPD 是什么

OPD 指 On-Policy Distillation。它不是离线收集一批教师完整答案再让学生模仿，而是：

```text
当前学生 Planner rollout
  -> 得到学生真实会进入的中间状态
  -> 教师模型针对这些状态给出更优 next_step / 停止判断 / 偏好
  -> 用这些纠偏数据训练学生
```

OPD 适合 AgentFlow，因为小 Planner 常见问题不是完全不会解题，而是在自己的错误上下文里不知道怎么恢复。on-policy 状态能覆盖这些真实错误分布。

### 9.3 RL 是什么

RL 阶段用最终任务 reward 优化 Planner。它不要求教师告诉每一步标准答案，而是通过多采样和 reward 比较，让模型偏向最终能答对的规划行为。

RL 的价值：

| 价值 | 说明 |
|---|---|
| 直接优化最终指标 | 目标和 AIME accuracy 更一致 |
| 保留探索 | 不完全受教师风格限制 |
| 能修正教师偏差 | 最终以 reward 为准 |
| 能发现非教师路径 | 多采样可能找到新的有效策略 |

## 10. SFT 常见问题与解决

| 问题 | 表现 | 原因 | 解决 |
|---|---|---|---|
| 格式学不会 | next_step 缺少 Tool Name 或 Sub-Goal | SFT 数据格式不统一 | 统一模板，过滤坏样本 |
| 学成答案生成器 | Planner 直接给最终答案，不调用工具 | 教师数据过多直接解题 | SFT 数据只监督规划 turn，不把工具结果当 Planner 输出 |
| 工具名不稳定 | 输出 `python`、`PythonCoder` 等别名 | 工具名没有固定枚举 | prompt 中列出精确工具名，SFT 只保留合法工具名 |
| 上下文太长 | 训练 OOM 或截断 | 把完整轨迹都塞进单样本 | 按 turn 训练，压缩 memory |
| 过拟合模板 | 换题后只套格式不思考 | SFT 数据太少或太单一 | 增加题型、工具类型、失败恢复样本 |
| final answer 格式差 | 没有 `\boxed{}` | SFT 未覆盖最终输出格式 | final_output 数据单独做格式强化 |
| SFT 后探索变差 | 输出过于确定 | 数据太干净、温度太低 | RL rollout 保留 temperature，SFT 只做热启动 |

## 11. OPD 常见问题与解决

| 问题 | 表现 | 原因 | 解决 |
|---|---|---|---|
| 教师直接把整题解完 | 学生学不到下一步规划 | prompt 没有限制教师角色 | 要求教师只修正当前 next_step 或给偏好，不直接终局代答 |
| 教师纠偏脱离学生状态 | 学生执行不了教师建议 | 教师没看到 memory 和工具结果 | OPD prompt 必须包含 question、analysis、memory、student_action、tool result |
| 成本太高 | API 调用量大 | 所有轨迹都送教师 | 优先采失败、低分、格式错、过早停止样本 |
| 教师偏好不可靠 | 蒸馏后指标不升 | 教师判断和 reward 不一致 | 用 reward_before/reward_after 或人工抽查过滤 |
| 学生只模仿教师风格 | RL 提升有限 | OPD 数据占比过高 | OPD 后继续 GRPO，保留采样探索 |
| on-policy 状态太差 | 教师纠偏样本混乱 | 原始 Planner 格式能力不足 | 先小规模 SFT 热启动 |
| 数据分布很快过期 | 后续策略变了 | OPD 只采了一轮 | 周期性采样：rollout -> teacher -> train -> rollout |
| 停止判断变保守 | 总是 CONTINUE 或总是 STOP | 教师偏好单一 | 构造 STOP/CONTINUE 平衡样本 |

OPD 数据建议字段：

| 字段 | 说明 |
|---|---|
| `question` | 原题 |
| `analysis` | 当前 Planner 初始分析 |
| `memory` | 已有工具调用和结果 |
| `student_action` | 学生原 next_step |
| `teacher_action` | 教师修正 next_step |
| `teacher_reason` | 教师解释 |
| `reward_before` | 原轨迹 reward |
| `reward_after` | 修正后或重放后的 reward |
| `action_type` | plan / next_step / stop |

## 12. RL 常见问题与解决

| 问题 | 表现 | 原因 | 解决 |
|---|---|---|---|
| reward 全是 0 | loss 没有效果，accuracy 不动 | 初始策略太差或 judge 太严 | SFT 热启动，降低任务难度，先小样本轨迹排查 |
| reward 全是 1 | advantage 接近 0 | 任务太简单或数据泄漏 | 换更难数据，检查 label 和 prompt |
| 组内 reward 方差低 | GRPO 学不动 | `n_samples_per_prompt` 太小或采样太保守 | 增大采样数，提高 temperature |
| KL 爆炸 | 输出退化、重复、格式坏 | 学习率过高或 KL 系数太低 | 降 lr，提高 `kl-loss-coef`，缩短 response |
| OOM | SGLang 或 trainer 崩 | context、batch、response 太大 | 降 `CTX_LEN`、`MAX_NEW_TOKENS`、`MAX_TOKENS_PER_GPU`、batch |
| 训练很慢 | rollout 等 API | 支持角色 API 延迟高 | 降并发/步数，换快模型，加缓存，先小样本 |
| 工具解析失败 | `Command parse error` | Executor 输出格式不合法 | 强化 Executor prompt，要求 triple quotes，过滤工具名 |
| 过早 STOP | 结果不完整 | Verifier 过宽或 Planner 子目标不清 | 降 max_steps 不解决，要看轨迹并改 Verifier prompt |
| 不停 CONTINUE | 步数耗尽 | Verifier 过严或 Planner 无停止意识 | 增加 STOP SFT/OPD 样本，限制 max_steps |
| 答案格式错 | reward 偏低 | final_output 没按 `\boxed{}` | 强化 final_output prompt，加入格式检查 |
| 训练后指标下降 | policy drift | RL 数据少、reward 噪声、过优化 | checkpoint 回滚，降低 lr，混合 SFT/OPD 数据 |

## 13. 系统工程常见问题

| 问题 | 排查命令 | 解决 |
|---|---|---|
| Ray 起不来 | `ray status` | `ray stop --force` 后重新启动 |
| 端口占用 | `lsof -i :30000` | 换 `PLANNER_PORT` 或停旧服务 |
| SGLang 启动 OOM | 看启动日志 | 降 `MEM_FRACTION`、`CTX_LEN`、TP |
| API key 未传入 Ray | 看 train log 环境 | 在启动前 export，确认 runtime env 里有变量 |
| tokenizer 不匹配 | 看 prompt 渲染异常 | `TOKENIZER_PATH` 与 Planner HF 模型保持一致 |
| checkpoint 不能评测 | SGLang 加载失败 | 先转 HF，确认 `config.json` 存在 |
| 结果 JSON 为空 | 看 Python traceback | 检查数据路径、输出目录权限、API 配置 |
| 评测和训练表现差异大 | 对比 eval 和 rollout 参数 | 对齐 temperature、max_steps、支持角色模型 |

## 14. 面试高频问题

### Q1：这个项目到底训练了什么？

训练的是 Planner 小模型。它负责初始分析和每一步工具规划，包括 Context、Sub-Goal、Tool Name。Executor、Verifier、Rewarder 默认是 API 支持角色，不参与参数更新。

### Q2：为什么不直接训练整个 agent？

因为工具执行、Verifier、Rewarder 都不是连续可微模块，而且多模型同时训练工程复杂度和显存成本都高。这里把可学习的策略决策集中到 Planner 上，其他角色作为环境和评估器，结构更清楚，也更适合少卡资源。

### Q3：RL 的 action 是什么？

可以理解为 Planner 在每个 turn 输出的文本决策。它不是离散环境里的单个 action id，而是自然语言形式的规划动作，包括选择工具、描述上下文和设定子目标。

### Q4：reward 怎么分配到多步决策？

reward 是 trajectory-level 的最终答案 reward。`custom_convert.py` 会把一条 trajectory 的多个 Planner turn 展开，每个 turn 共享同一条轨迹的归一化 reward，并除以 turn 数，近似优化整条 agent flow 的平均决策质量。

### Q5：为什么 final_output 不训练？

当前设计里 final_output 由固定 API 模型生成，目的是把最终表达能力稳定下来，让 RL 主要优化 Planner 的规划能力。为了避免 API final_output 绕过工具直接重解题，现在 prompt 明确限制它只能汇总 Memory 中已有的工具结果；如果信息不足，应输出 `\boxed{INSUFFICIENT_TOOL_RESULTS}`。如果把 final_output 也纳入训练，需要保证它来自本地 Planner 并有正确 logprob，否则 loss 会不合法。

### Q6：Verifier 会不会影响训练？

会影响数据分布。Verifier 决定 STOP/CONTINUE，从而影响 trajectory 长度和最终答案质量。但 Verifier 本身不被训练，它相当于环境的一部分。

### Q7：为什么用 API Rewarder，而不是规则 reward？

数学答案存在等价形式，例如分数、小数、化简表达式。API Rewarder 可以先做语义等价判断，减少纯字符串匹配误判。更严谨的版本可以引入 sympy 或规则判分，降低 LLM judge 噪声。

### Q8：GRPO 相比 PPO 的优势是什么？

GRPO 不依赖单独 value model，而是用同 prompt 多样本的组内 reward 归一化估计 advantage，工程上更轻量，适合大模型 RLHF/RLAIF 场景。这个项目资源有限，所以 GRPO 比训练额外 critic 更合适。

### Q9：SFT 和 OPD 的区别是什么？

SFT 通常监督固定数据里的标准输出，可能是离线教师轨迹。OPD 是让学生当前策略先 rollout，再让教师在学生真实访问到的状态上纠偏。OPD 更关注分布偏移和错误恢复。

### Q10：为什么 OPD 适合这个项目？

因为小 Planner 的错误经常出现在自己的中间状态里，比如工具结果没吸收、错误停止、上下文缺失。离线教师完整轨迹很干净，但覆盖不到学生真实错误分布。OPD 能让教师针对这些 on-policy 状态给出更有训练价值的纠偏。

### Q11：如果 RL 训练没有提升，你怎么排查？

先看 reward 分布是否有方差，再看轨迹中的失败类型。具体会检查：格式是否可解析、工具是否执行成功、Verifier 是否过早停止、final_output 是否有 `\boxed{}`、Rewarder 是否误判、GRPO 组内 reward 是否全 0 或全 1。然后针对问题决定做 SFT、OPD、调采样温度、改 reward 或降 batch/context。

### Q12：如何证明提升来自 Planner，而不是 API 模型？

做对照实验：固定 Executor、Verifier、Rewarder 的 API 模型，只替换 Planner checkpoint。比较原始小 Planner、SFT Planner、OPD Planner、GRPO Planner、API Planner 上限。如果支持角色不变，指标变化主要来自 Planner。

### Q13：如何避免 Reward hacking？

不要只看 LLM judge 分数，要保留规则抽取、人工抽查、AIME 准确率和轨迹检查。对于可疑样本，看 final_output 是否真的解题，还是只学会输出形式。必要时引入更严格 judge 或 symbolic checker。

### Q14：max_steps 怎么选？

太小会导致复杂题没机会调用足够工具，太大会增加 API 成本和错误累积。单卡 baseline 可用 2 到 3，正式评测常用 5。应结合轨迹看 STOP/CONTINUE 行为和平均步数。

### Q15：为什么 Executor 还要由 LLM 生成 command？

Planner 输出的是自然语言的工具选择和子目标，Executor 把它转成统一工具接口 `tool.execute(query=...)`。这样 Planner 不需要学 Python 调用细节，只学规划语义，降低训练难度。

## 15. 轨迹分析模板

分析失败样本时按这个顺序看：

1. `analysis` 是否理解题目。
2. 第一个 `next_step` 是否选择了合理工具。
3. `Context` 是否包含工具需要的信息。
4. `tool_command` 是否可解析。
5. `execution_result` 是否真的有用。
6. Verifier 是过早 STOP 还是过度 CONTINUE。
7. `final_output` 是否吸收了工具结果。
8. `\boxed{}` 中的答案是否和 label 等价。
9. Rewarder 判分是否合理。

失败类型可以这样归类：

| 类型 | 说明 | 优先处理方式 |
|---|---|---|
| Planner 理解错题 | analysis 一开始就错 | SFT/OPD 题目理解样本 |
| 工具选错 | 应计算却选通用生成，或反之 | OPD next_step 纠偏 |
| Context 缺失 | 工具拿不到必要条件 | SFT 格式和上下文监督 |
| Command 解析失败 | Executor 输出非法 Python | 改 Executor prompt 或工具接口 |
| 工具结果没吸收 | 下一步忽略 execution_result | OPD memory 使用样本 |
| Verifier 过早停止 | 信息不够就 STOP | Verifier prompt 或 STOP/CONTINUE 数据 |
| final_output 错 | 工具结果对但汇总错 | 改 final_output 模型/prompt |
| Rewarder 误判 | 明明等价判错 | 加规则判分或人工复核 |

## 16. 推荐答辩叙事

可以按下面顺序讲：

1. 我先把 agent 拆成 Planner、Executor、Verifier、Rewarder。
2. 资源有限，所以只训练 Planner，把其他角色作为固定环境。
3. Planner 的每个 plan/next_step turn 都会记录 token 和 logprob。
4. 一条完整 trajectory 得到最终答案 reward。
5. `custom_convert` 把多轮 trajectory 展开成多个 Planner turn，用同一个归一化 reward 更新。
6. SFT 用来解决格式和工具协议冷启动。
7. OPD 用来让教师纠偏学生真实 on-policy 错误状态。
8. GRPO 用最终任务 reward 继续优化，不完全依赖教师。
9. 评测上做单轮 baseline、API Planner 上限、本地 Planner RL 前后对照。
10. 最后通过轨迹说明提升来自更好的工具选择、停止判断和答案整合。

## 17. 最重要的面试金句

| 场景 | 可以这样说 |
|---|---|
| 解释 OPD | 我不是离线蒸馏教师答案，而是在学生当前策略访问到的状态上让教师纠偏，解决分布偏移。 |
| 解释 GRPO | 我对同一题采多条轨迹，用组内 reward 差异给 Planner turn 分配 advantage。 |
| 解释只训 Planner | 我把支持角色固定为环境，把可学习决策集中在 Planner，降低显存和 credit assignment 难度。 |
| 解释失败排查 | 我先看 reward 分布，再看轨迹，把失败分成理解、工具选择、执行、停止、汇总、判分六类。 |
| 解释 baseline | 我用单轮 QA、本地 AgentFlow、API Planner 三组对照拆分基础模型能力、系统能力和教师上限。 |

## 18. 新手版：先分清“模型、环境、训练框架”

很多人在面试里讲不清，是因为把所有东西都叫模型。这个项目里至少有三层：

| 层 | 包含什么 | 是否训练 |
|---|---|---:|
| 被训练模型 | 本地 Planner | 是 |
| Agent 环境 | Executor、Tool、Memory、Verifier、final_output、Rewarder | 否 |
| 训练框架 | slime、Ray、SGLang、custom rollout、custom convert | 否 |

你可以这样理解：

```text
Planner 是学生
Agent 环境是考试场景和工具
Rewarder 是阅卷老师
slime/GRPO 是训练方法
```

面试中要避免说“我训练了整个 AgentFlow”。更准确的是：

> 我训练的是 AgentFlow 里的 Planner 策略模型，其他模块作为固定环境和支持角色。

## 19. GRPO 更细的直觉解释

假设同一道题采 4 条轨迹：

| 轨迹 | Planner 行为 | reward |
|---|---|---:|
| A | 选对工具，最终答对 | 1 |
| B | 工具选错，答错 | 0 |
| C | 过早停止，答错 | 0 |
| D | 多走一步但答对 | 1 |

GRPO 会在这组内部比较。它不是简单地说 reward=1 就永远好，而是看同一题不同输出的相对表现。

直觉：

```text
同一题里答对的轨迹 -> 提高概率
同一题里答错的轨迹 -> 降低概率
```

放到 AgentFlow：

```text
轨迹 A 的 plan / next_step turns 都会得到正向信号
轨迹 B/C 的 plan / next_step turns 得到负向信号
轨迹 D 也得到正向信号，但如果它步数过长，后续可以用长度惩罚或分析平均 step 控制
```

重要限制：

| 限制 | 解释 |
|---|---|
| reward 是稀疏的 | 只有最终答案对错，不知道哪一步错 |
| credit assignment 不完美 | 答错可能是某一步错，也可能是 final_output 错 |
| 需要多样性 | 如果同题采样都一样，GRPO 学不到差异 |
| reward 噪声会放大 | judge 错了会给错误方向的信号 |

所以你需要轨迹分析、SFT/OPD 热启动和 reward 质量控制。

## 20. SFT 数据应该长什么样

SFT 不一定要训练完整 answer。对于这个项目，最有价值的是训练 Planner 的格式和决策。

### 20.1 plan 样本

输入：

```text
Task: Analyze the given query...
Query: <数学题>
Available tools: [...]
Metadata: {...}
```

目标输出：

```text
这道题需要先识别核心关系，再用局部数学推理工具检查一个关键公式，必要时用 Python 做枚举或数值验证。需要避免直接跳最终答案。
```

### 20.2 next_step 样本

输入包含：

```text
Query
Query Analysis
Available Tools
Previous Steps
```

目标输出必须稳定：

```text
Justification: ...
Context: ...
Sub-Goal: ...
Tool Name: Python_Code_Generator_Tool
```

过滤规则：

| 坏样本 | 为什么过滤 |
|---|---|
| 缺少 Tool Name | parser 解析不了 |
| Tool Name 不在工具列表 | 默认会工具加载失败；只有旧 checkpoint 评测才开启 alias 兼容 |
| Sub-Goal 太大 | 工具变成直接解题 |
| Context 没有必要信息 | Executor 不知道怎么写 query |
| 把工具结果当 Planner 输出 | loss 会教错角色 |

## 21. OPD 的具体落地流程

一个可落地的小规模 OPD 流程：

```text
1. 用当前 Planner 跑 20 条题
2. 保存 trajectory
3. 找出失败轨迹或低质量轨迹
4. 对每条轨迹截取某个中间状态
5. 发给教师模型，让教师只修正下一步
6. 把教师 next_step 转成 SFT/偏好数据
7. 训练 Planner
8. 再跑同类题，看工具选择和 STOP 是否改善
```

教师 prompt 应该强调：

| 要求 | 原因 |
|---|---|
| 不要直接给最终答案 | 避免学生绕过规划 |
| 只修正当前一步 | 保持 on-policy 状态对应 |
| 使用已有 memory | 避免教师忽略学生当前上下文 |
| 输出合法工具名 | 保证可执行 |
| 说明为什么更好 | 可用于偏好数据或人工检查 |

教师输入可以包含：

```text
Question:
...

Student analysis:
...

Memory so far:
...

Student next_step:
...

Tool list:
...

Please produce a better next_step only.
```

## 22. RL 问题排查的决策树

如果训练没提升，按这个顺序排查：

```text
accuracy 不升
  -> reward 有没有方差？
      -> 没有：看是否全 0 或全 1
      -> 有：继续
  -> Planner 输出能解析吗？
      -> 不能：先 SFT 格式
      -> 能：继续
  -> 工具调用成功吗？
      -> 不成功：看 Executor command 和 tool_name
      -> 成功：继续
  -> Verifier 停止合理吗？
      -> 过早/过晚：改 Verifier 或做 OPD STOP 数据
      -> 合理：继续
  -> final_output 是否正确吸收 memory？
      -> 否：换 final_output 模型或 prompt
      -> 是：继续
  -> Rewarder 是否误判？
      -> 是：加规则判分
      -> 否：调 GRPO 超参或扩大数据
```

常见优先级：

| 优先级 | 问题 | 原因 |
|---|---|---|
| 最高 | 格式不可解析 | 后面全断 |
| 高 | reward 全 0 | 没有学习信号 |
| 高 | 工具调用失败 | Agent 链路不成立 |
| 中 | Verifier 判断差 | 影响步数和成本 |
| 中 | final_output 格式差 | 影响 reward |
| 低 | 小幅指标波动 | 小样本正常现象 |

## 23. 面试追问：如果让你继续改进项目

可以从低风险到高风险讲：

| 改进方向 | 做法 | 价值 |
|---|---|---|
| 规则 reward | 加 sympy/数学答案 normalize | 降低 LLM judge 噪声 |
| 更好的轨迹日志 | 统计解析失败、STOP、工具成功率 | 更容易定位问题 |
| SFT 热启动 | 训练合法格式和工具协议 | 降低 RL 冷启动难度 |
| OPD 数据闭环 | 教师纠偏学生失败状态 | 解决分布偏移 |
| 工具 schema 化 | 用结构化 JSON 代替自由文本 | 降低 parser 错误 |
| Verifier 单独训练 | 收集 STOP/CONTINUE 数据 | 降低 API 依赖 |
| final_output 本地化 | 让 Planner 或小模型生成答案 | 更完整端到端训练 |

注意面试时不要一上来就说“全都端到端训练”。更稳妥的回答是：

> 我会先把当前链路中错误率最高的环节数据化，比如工具解析失败率、STOP 错误率和 Rewarder 误判率。先做可观测性和 SFT/OPD，再考虑更复杂的端到端训练。

## 24. 更长的面试自我介绍版本

可以这样讲一段完整项目经历：

> 我做的是一个基于 slime 的 AgentFlow 复现和训练项目。系统里我把 Agent 拆成 Planner、Executor、Tool、Verifier、Rewarder 几个角色。由于本地 GPU 有限，我只把小模型 Planner 放在本地训练，其他角色默认走 OpenAI 兼容 API。一次 rollout 中，Planner 先分析题目，再根据 Memory 选择下一步工具和子目标；Executor 把它转成统一的 `tool.execute(query=...)` 调用；Tool 返回局部结果；Verifier 决定继续还是停止；最后 final_output 汇总答案，Rewarder 根据 label 给 0/1 分。训练上我用 GRPO，同一题采多条轨迹，通过组内 reward 差异优化 Planner。因为一条轨迹有多个 Planner turn，我写了 custom_convert，把 plan 和 next_step 拆成独立训练样本，并让它们共享 trajectory-level reward。为了提高冷启动稳定性，我会先考虑 SFT 让 Planner 学会工具协议，再用 OPD 在学生真实 on-policy 错误状态上让教师纠偏，最后接 GRPO 优化最终指标。
