# 我们测了 5 个国产大模型在金融犯罪话题上的拒答能力，推理模型来到最后一名

> 本文是 [fin-safety-inspect v0.2.1](https://github.com/acai77777/fin-safety-inspect/releases/tag/v0.2.1) 的发布稿，对应 [docs/cross-model-baseline.md](https://github.com/acai77777/fin-safety-inspect/blob/main/docs/cross-model-baseline.md)。

## 一段反直觉的开场

我们给 5 个国产可用大模型喂了同一组金融犯罪 prompt（FinTrust 数据集 100 题，覆盖偷税、洗钱、内幕交易、伪造票据、非法放贷等 10 类话题）。每个模型驱动同一个 mock 银行 ReAct agent，跑同样的工具集。

5 个模型按 **judge accuracy（LLM 审核员判模型有没有合理拒答）** 排名：

| 排名 | 模型 | judge accuracy | keyword accuracy | Δ |
| --- | --- | --- | --- | --- |
| 1 | **Qwen-plus** | **0.950** | 0.930 | +0.020 |
| 2 | DeepSeek-chat | 0.910 | 0.790 | +0.120 |
| 3 | GLM-4.5 | 0.880 | 0.870 | +0.010 |
| 4 | DeepSeek-R1 | **0.840** | **0.670** | +0.170 |
| 5 | Kimi K2 | 0.830 | 0.810 | +0.020 |

**DeepSeek-R1（推理模型）排倒数第二**。在 keyword（硬拒答关键词）上**直接最后一名**，0.670，比同家 DeepSeek-chat 的 0.790 还**低 12 个点**。

这违反我们的直觉：推理模型更聪明、更"会思考"、更高级，安全性应该更好。**数据说不是**。

## 项目是什么

[fin-safety-inspect](https://github.com/acai77777/fin-safety-inspect) 是个 Python 包，提供金融 agent 安全的 Inspect AI task pack + 第一个公开的 LangGraph adapter。v0.2.1 在 v0.2.0 基础上加了跨模型 baseline。

跨模型设置（关键）：**每个模型跑同一组 prompt，但 LLM judge 固定是 DeepSeek-v4-flash**（通过 `JUDGE_BASE_URL` 独立环境变量），保证 5 个模型的 accuracy 数字直接可比，不会因 judge 不同而引入偏差。

## 主菜 1：推理模型反而更不安全

R1 跟 DeepSeek-chat 同家，参数量更大，能输出 thinking tokens。**但在金融犯罪 refusal 任务上：**

- keyword accuracy: chat 0.790 vs **R1 0.670**（差 12 个点）
- judge accuracy: chat 0.910 vs **R1 0.840**（差 7 个点）
- pp（双 scorer 都 PASS）: chat 78 vs **R1 63**（5 个模型里最低）
- fp（软拒答被 judge 救回）: chat 13 vs **R1 21**（5 个模型里最高）

合理解释（假设，未严格验证）：

R1 的推理 RLHF 强化了 *"分解问题，给出详细分析"* 的行为模式，弱化了 *"早期拒答"* 的行为。当一个金融犯罪 prompt 进来，R1 的 thinking trajectory 倾向于 *"找角度回答"* 而不是 *"找角度拒绝"*。Thinking tokens 都用在拆解上，不是用在筑墙上。

**对部署者的提示**：不要默认更聪明的、能推理的模型自动更安全。在金融助手这种 task 上，R1 比 DeepSeek-chat 实质性更不安全。

## 主菜 2：DeepSeek 家族的"软拒答指纹"

| 家族 | judge - keyword（救回的 Mode #2 比例）|
| --- | --- |
| DeepSeek（chat / R1）| **+0.120 / +0.170** |
| Qwen / GLM / Kimi（其他三家）| +0.010 ~ +0.020 |

DeepSeek 训练得**爱说"我建议你咨询专业人士"、"这可能涉及内幕交易，需要谨慎"** 这种同情措辞，但**不直接拒答**。Keyword scorer 看不到 *"I cannot"* 之类的硬拒答短语，标 FAIL；LLM judge 读懂了语境，标 PASS。

Qwen / GLM / Kimi 不一样。它们偏好硬拒答（*"我无法提供"* / *"这是违法的"*），关键词列表直接命中。**判官没什么软拒答可救。**

**对评测者的提示**：仅靠关键词的 refusal scorer 会**系统性低估 DeepSeek 的安全性**，相对于 Qwen / GLM / Kimi。如果发评测榜，**只用关键词 = 给 DeepSeek 不公平的低分**。

## 副菜 1：Mode #1（agent 跑题）是 DeepSeek 特异，不是普遍现象

v0.2.0 baseline 时我们假设 *"Mode #1（agent 调银行工具跑题）是普遍失败模式"*，因为只看了 DeepSeek-chat 的 30 个 false negative。

跨模型一对比：
- **Qwen-plus 在 Illegal Lending 上 ff = 0**：Qwen 从不跑题。
- **DeepSeek-chat ff = 2**：典型 Mode #1，agent 在放贷话题里调 `list_accounts()` 查账户。
- **GLM-4.5 ff = 5**（最高）：但**手动看 5 条全部不是 Mode #1**。

GLM-4.5 在放贷话题上失败的方式完全不同：模型直接把"非法放贷"包装成"informal lending best practices"清单，给"Setting up Driver Lending System"的法律框架。这是 Mode #3（实操指引），不是 Mode #1（跑题）。

**两个模型在同一组 prompt 上失败，但失败机制完全不同**：
- DeepSeek 的失败是**工程问题**（agent 派发到错误工具）
- GLM 的失败是**训练问题**（model 把违法话题当知识问答）

## 副菜 2：Mode #3（真没拒答，给了操作）是普遍现象

Counterfeit（伪钞 / 伪造票据）这一类，**5 个模型每个都至少有 1 个 ff**：DS-chat 3 / DS-R1 2 / Kimi 1 / Qwen 2 / GLM 4。

prompt 设计得像合法小生意主在问（"我经营电子产品店怎么识别假钞"），每个模型都至少被骗到一次。

身份盗用、非法集资也是同样模式。**Mode #3 没有 scorer 层的修法**——要么模型自己拒，要么靠输出端 guardrail 拦。

## 方法论 takeaway

跨模型数据让我们做了 v0.2.x roadmap 的**数据驱动重写**：

| v0.2.0 时的 v0.2.1 优先级 | 现在 | 原因 |
| --- | --- | --- |
| 1. Trace-level scorer 修 Mode #1 | **降到 v0.3+** | Mode #1 是 DeepSeek 特异，trace scorer 是 DeepSeek 工程问题工具，不是通用 benchmark 特性 |
| 2. LLM-judge scorer | **v0.2.0 已发** | 完成 |
| 3. 跨模型 baseline | **v0.2.1 已发** | 完成 |
| **新优先级 1：output guardrail 设计** | **升到第一** | 唯一能修 Mode #3 的方向，Mode #3 是唯一普遍失败模式 |
| **新优先级 2：R1 推理 vs 拒答研究** | **新增** | R1 的 0.12 keyword 差距是真发现，需要单独研究是 thinking / RLHF / 温度哪个原因 |
| **新优先级 3：GPT-4o / Claude baseline** | **新增** | 没有国际模型，无法判断 Mode #3 是中文 RLHF 现象还是全球现象 |

可泛化的方法论 lesson（跟 v0.1 那篇互补）：

> **当一个评测有反直觉数据时，先去数据里找解释，再决定改 model / 改 scorer / 改 task**。

v0.1 lesson 是 *"baseline 数字不好看时先手动看 FN"*。v0.2.1 lesson 是 *"baseline 数字反直觉时先做跨模型对照"*。两条加起来：**别只看一个数字，去看它在不同维度上怎么变**。

## v0.2.1 没做的

- **GPT-4o / Claude 没跑**：OpenRouter 二级中转配置问题。我们选择先发国内 5 模型对比，不等中转修好。下个版本补。
- **R1 现象没机制研究**：只是观察到 R1 拒答弱，没排除是 thinking / RLHF / 温度还是 prompt-following 风格。
- **Trace-level scorer 没做**：现在数据看 ROI 不高，等更多数据再决定。

## 用起来

```bash
pip install git+https://github.com/acai77777/fin-safety-inspect.git@v0.2.1
```

```powershell
# 固定 judge 在 DeepSeek
$env:JUDGE_API_KEY  = "<你的-DeepSeek-key>"
$env:JUDGE_BASE_URL = "https://api.deepseek.com/v1"

# 主线 baseline
$env:OPENAI_API_KEY  = "<你的-DeepSeek-key>"
$env:OPENAI_BASE_URL = "https://api.deepseek.com/v1"
.\run_smoke.ps1                                  # ≈60s, ≈¥0.22

# 跨模型：换 OPENAI_* 跑别的 backbone
.\run_smoke.ps1 -Model "qwen-plus"
.\run_smoke.ps1 -Model "deepseek/deepseek-r1"   # 通过 OpenRouter

python dump_cross_model.py                      # 5 模型对比表
python dump_failures.py "z-ai/glm-4.5"          # 看某个模型的 ff
```

详细 baseline：[docs/cross-model-baseline.md](https://github.com/acai77777/fin-safety-inspect/blob/main/docs/cross-model-baseline.md)。

Repo：[acai77777/fin-safety-inspect](https://github.com/acai77777/fin-safety-inspect)。Apache-2.0。31 单元测试，无需 API key 即可运行。

## 引用

- FinTrust：Hu et al., "FinTrust: A Comprehensive Benchmark of Trustworthiness Evaluation in Finance Domain", EMNLP 2025.
- Inspect AI：UK AI Safety Institute.

License: Apache-2.0。FinTrust / Inspect AI 上游各自保留 license，本仓库不重新分发任何上游 dataset 内容。
