# 情感长视频推理 Benchmark 构建流程方案

## Context

"情感长视频推理"三篇论文中 **Benchmark 篇** 的 pipeline 设计。

**任务定位**：构建一个面向 **情感领域长程视频推理** 的 benchmark，覆盖 9 个具体任务（识别 / 多模态归因 / 时序定位 / 因果归因 / 多跳因果 / 短期行动预测 / 跨集行动预测 / 个性一致性预测 / 情感影响他人），用以评估：
- 端到端 MLLM（Qwen2.5-Omni、GPT-5、Gemini 2.5、InternVL3 等）
- Video Agent 方法（含本三篇论文中的另两篇 + M3-Agent 等外部 baseline）

**已对齐决策**（用户已确认）：
| 维度 | 决策 |
|---|---|
| Memory 组件 | **M1–M7 全集**（局部感知 / 集内事件图 / 跨集事件图 / 角色情感轨迹 / 角色人格档案 / 关系网络 / 世界状态） |
| Lifecycle 范围 | **单季全集**（中英各 3 部 × 首季 10 集 = 60 集） |
| 评估 Setting | **E0+E1+E2+E3 四档**（局部片段 / 整集 / 全季 / 显式语义记忆） |
| 任务集 | **9 个**：T1 T2 T4 T5 T6 T7 T8 T9 T10（含多跳因果，无隐藏情感） |
| Predictive horizon | 多档并行：**1min / 5min / 30min / cross-episode** |
| 自动化程度 | **全自动**，仅 Stage 8 人类对照为人工 |
| 语种 | **中英双语并行**，分层采样 |
| 模型分工 | **Codex 全权处理**：GPT-5 多模态负责视觉/事件抽取/语义沉淀/QC + 编排，**Qwen2.5-Omni** 由 Codex 调度处理音视频联合 + 副语言 |
| 终版规模 | ~1,800 QA（9 任务 × 60 集 × 平均 3-4 题/任务/集，过滤后） |

---

## 两大设计轴：Memory 组件 (M) × 评估 Setting (E)

Benchmark 的两大正交设计维度。M 决定 **pipeline 出题时锚定到多丰富的信息**；E 决定 **评估时给被测模型多少 context**。9 个任务在这两个轴上分布形成评估矩阵。

---

### 设计轴 A：Memory 组件 (M1–M7)

#### 7 个 Memory 组件

| 代号 | 组件 | 内容 | 抽取阶段 | 粒度 |
|---|---|---|---|---|
| M1 | **局部感知** | 当前事件的视/听/对白 caption + 微表情/副语言 | Stage 1 | shot/event |
| M2 | **集内事件图** | 单集 N 事件 + DAG（causal/trigger/predicts_action 等） | Stage 2-3 | episode |
| M3 | **跨集事件图** | 全季事件 DAG（M2 跨集 merge） | Stage 4.5 | season |
| M4 | **角色情感轨迹** | 角色情感曲线（含跨集累积） | Stage 4 + 4.5 | per-char × time |
| M5 | **角色人格档案** | OCEAN + 关键背景 + 行为模式（语义沉淀） | Stage 4.6 | per-char |
| M6 | **关系网络** | 角色关系类型 + 演化 | Stage 4.6 | char × char |
| M7 | **世界/剧情状态** | 关键剧情设定与时间演化 | Stage 4.6 | season |

### 9 个任务 × Memory 依赖

| # | 任务 | 一句话题目示例 | 必需 M | 可选 M | 主测能力 |
|---|---|---|---|---|---|
| T1 | 情感识别 | "ev07 时 Walter 的情感是？" | M1 | — | 多模态分类 |
| T2 | 多模态证据归因 | "下列哪条线索最能说明 Walter 此刻的内疚？" (4 选 1 跨模态) | M1 | — | 细粒度感知 |
| T4 | 情感转折定位 | "Walter 从平静转为愤怒发生在哪个时刻/事件？" | M1+M2 | M3 | 时序定位 |
| T5 | 情感原因归因 | "Walter 在 ev07 内疚的最直接诱因是？" | M2 | M3, M5 | 单跳因果 |
| T6 | 多跳因果链 | "导致 Walter 此刻崩溃的事件链是？" (3+ 跳) | M2/M3 | M5 | 长链因果 |
| T7 | 短期行动预测 | "ev07 后 5 min 内 Walter 最可能做什么？" | M2 (future GT) | M5 | 情感→近期行动 |
| T8 | 跨集行动预测 | "Walter 在 S01E03 的羞辱将在哪集引发何种行动？" | **M3+M4** | M5 | 情感→长程后果 |
| T9 | 个性一致性预测 | "基于 Walter 的人格，面对此次羞辱他会选择？" | **M5** | M6 | 个性 × 情境 |
| T10 | 情感影响他人 | "Walter 摔门后，Skyler 的情感与行为如何变化？" | **M4×N + M6** | — | 人际情感传染 |

> **解读**：T1-T7 单集可答（≤M2），T8-T10 必须跨集 + 语义记忆。这是 benchmark 长程依赖密度的来源。

---

### 设计轴 B：Evaluation Setting (E0–E3)

控制**被测模型**能看到多少 context 的维度，用以量化长程依赖能力。

#### 一图速览

| 代号 | 给模型的输入 | 测试目标 |
|---|---|---|
| E0 | 仅问题相关 2-5min 片段 | 局部多模态能力（baseline） |
| E1 | 完整当集 | 集内长上下文 |
| E2 | 全季视频 / 全季事件摘要 | 跨集长程记忆 |
| E3 | 全季 + 显式提供 M5/M6/M7 | 测"有记忆系统"的 agent 能不能用好语义记忆 |

#### 详解

**E0：仅局部片段（~2-5 min）**
- 怎么喂：截取问题对应事件前后 2-5min 的视频片段
- 测什么：纯多模态感知，无长程记忆
- 必选理由：基线，没它就无法量化长程增益
- 预期：T1/T2 可答；T4-T7 部分可答；T6/T8/T9/T10 期望失败

**E1：完整当集（~45 min）**
- 怎么喂：整集 mp4（或抽帧+音频）；现代 MLLM 都吃得下
- 测什么：集内长上下文能力
- 作用：与 E0 对比 → 量化集内长上下文增益
- 预期：T1-T7 应可答；T8/T9/T10 仍失败

**E2：全季视频/摘要（~10 集 ≈ 7-10 h）**

没有模型能直接吃 10h 视频，两种实现：
- **E2-raw**：抽帧到 0.1fps + ASR 文本，整季塞 1M context 模型（GPT-5 / Gemini 2.5 Pro）。优：保留原始信号。缺：工程难、API 贵。
- **E2-summary**：前 N-1 集 LLM 生成 1-2k tokens 摘要 + 当集完整视频。优：成本低。缺：测的是"理解摘要"而非"记忆"。

- 测什么：跨集记忆能力（M3/M4 出题真正发挥的地方）
- 预期：T8 在 E2 应出现明显增益；T9/T10 部分可答

**E3：显式语义记忆（M5+M6+M7 喂入）**
- 怎么喂：pipeline 抽好的 M5（人格档案）+ M6（关系）+ M7（世界状态）作为结构化 context（JSON/Markdown）+ 当集完整视频
- 测什么：模型能否**用好已经存在的语义记忆**——Video Agent 的核心能力
- 关键作用：
  - 与 E1 对比 → 量化"有语义记忆 vs 没有"的增益
  - 与 E2 对比 → 量化"显式结构化记忆 vs 模型自看全季压缩"哪个更有效
  - **与另两篇 Video Agent 协同评估的关键档**：Agent 自带记忆系统 vs MLLM 用我们注入的记忆
- 预期：T9 在 E3 应出现最大增益；T8/T10 也应明显改善

#### 4 档关系

```
E0 (片段)   →   E1 (当集)   →   E2 (全季)   →   E3 (显式记忆)
   │              │               │                │
 纯感知        集内长程        跨集长程         Agent 记忆能力
   │              │               │                │
基线           ↑ 集内增益     ↑ 跨集增益       ↑ 语义记忆增益
```

#### 实现成本

| Setting | 单条评估 token 量 | API 成本（相对） | 工程难度 |
|---|---|---|---|
| E0 | 1× | 1× | 低 |
| E1 | 5-10× | 5-10× | 低 |
| E2-raw | 50-100× | 50-100× | 高 |
| E2-summary | 8-15× | 8-15× | 中 |
| E3 | 6-12× | 6-12× | 中 |

---

### 9 个任务 × Evaluation Setting 期望表现

| 任务 | E0 局部 | E1 当集 | E2 全季 | E3 显式记忆 |
|---|:---:|:---:|:---:|:---:|
| T1 | ★ | ★ | ★ | ★ |
| T2 | ★ | ★ | ★ | ★ |
| T4 | ⚠ | ★ | ★ | ★ |
| T5 | ⚠ | ★ | ★ | ★ |
| T6 | ✗ | ⚠ | ★ | ★ |
| T7 | ⚠ | ★ | ★ | ★ |
| T8 | ✗ | ✗ | ★ | ★★ |
| T9 | ✗ | ⚠ | ⚠ | ★★ |
| T10 | ✗ | ★ | ★ | ★★ |

★ 期望可答 / ⚠ 部分可答 / ✗ 期望失败。**T8/T9 是区分"有无记忆系统"的核心任务**，与另两篇 Video Agent 论文协同评估。

---

## Pipeline 总览（事件中心 + 跨集 merge + 语义沉淀）

```
[Stage 0] 单集视频采集（60 集）
      │
      ▼
[Stage 1] 多模态结构化感知（每集独立，Codex 统一编排）
            GPT-5(视觉) | Qwen-Omni(音/副语言) | PySceneDetect | RetinaFace
      │
      ▼
[Stage 2] 事件抽取（affect / action / social）
      │
      ▼
[Stage 3] 集内事件关系图 DAG → M2
      │
      ▼
[Stage 4] 集内角色情感轨迹 → M4(per-episode)
      │
      ▼
[Stage 4.5] 跨集 merge → M3 跨集事件图 + M4 累积轨迹
      │
      ▼
[Stage 4.6] 语义记忆沉淀 → M5 人格 + M6 关系 + M7 世界
      │
      ▼
[Stage 5] 9 任务族问题生成（按任务-M 依赖矩阵勾选 context）
      │
      ▼
[Stage 6] 6 道自动质量过滤
      │
      ▼
[Stage 7] 评估面板：MLLM × {E0,E1,E2,E3} + Video Agent
      │
      ▼
[Stage 8] 人类对照实验（唯一人工环节）
```

---

## Stage 1：多模态结构化感知（单集）

**输入**：
- `episode.mp4` — 一集视频文件（必需）
- `.srt` — 字幕文件，时间戳 + 对白文本（可选；有就跳过 ASR，没有就用 Qwen2.5-Omni 听）

**输出**：**时间对齐 multi-track perception 文件**——把同一段视频从多角度抽出来的信息，全部按时间戳对齐到一个 JSON。"多轨"类比剪辑软件的轨道概念（视频轨/音频轨/字幕轨），各跑各的但能横向对齐。任何一个时间点 `t` 都能同时查到该时刻：当前画面中谁、他/她的表情和动作、说了什么、语调如何、是第几个 shot。Stage 2 抽事件时直接从 perception 文件读取多模态信息。

所有 track 由 **Codex** 统一编排（Bash / Python 子任务由 Codex 调度执行）。

| Track | 模型 / 工具 | 说明 |
|---|---|---|
| Shot boundary | PySceneDetect (ContentDetector) | **镜头边界检测**：检测视频中"切镜头"的位置（如 Walter 特写 → Skyler 反应特写），把视频自动切成 1-30s 的 shot 作为最小处理单元。PySceneDetect 是社区标准开源库；ContentDetector 是其中基于 HSV 像素差的算法，对剧集场景切换最稳定 |
| Visual caption + 微表情/肢体 | **GPT-5 多模态**，0.5fps 抽帧 + 关键帧 burst | 每个 shot 生成视觉描述，关键帧额外问微表情/姿态 |
| ASR + utterance 时间戳 | Qwen2.5-Omni（中文优先 SenseVoice 备选） | 把对白转成"几点几秒说了什么" |
| 副语言（prosody） | Qwen2.5-Omni 显式问"语调/停顿/颤抖/音量" | 不是听内容，是听**怎么说**——情感感知的关键线索 |
| Speaker diarization | pyannote-audio | 把音频按说话人分簇，回答"这句话是谁说的" |
| 人脸检测+聚类 | RetinaFace + ArcFace + 层次聚类 | 找出每帧的人脸，跨帧聚类成同一角色 |
| 角色命名对齐 | GPT-5 用字幕"被称呼的名字" × face cluster 交叉 | 把人脸聚类（face_cluster_03）对齐到具体角色名（"Walter"） |

输出 schema：`data/perception/<series>/<episode>.jsonl`（shot/event-level 多 track 字段）

---

## Stage 2：事件抽取

**事件定义**：有明确起止时间 + 主体 + 类型 ∈ {affect_event, action_event, social_event}

**方法**：GPT-5（吃完整集，长上下文）+ Qwen2.5-Omni（补 affect 信号）双 pass → IoU + 语义合并

每个事件字段：
```jsonc
{ "event_id": "ep03_ev07",
  "type": "affect_event",
  "time_span": [873.2, 891.5],
  "participants": ["Walter"],
  "emotion": "guilt", "intensity": 0.8,
  "trigger_ref": "ep03_ev05",
  "cues": [{"modality":"visual","ts":[878,882],"desc":"avert eyes"},
           {"modality":"audio","ts":[886,891],"desc":"voice tremor"}],
  "summary": "..." }
```

---

## Stage 3：集内事件关系图 → M2

GPT-5 + Qwen2.5-Omni 双方各出关系候选 → 取交集为高置信边，并集为候选边（人工不介入，靠 Stage 6 过滤）。

边类型：`temporal_after / causal / emotion_trigger / counter_action / reveals_belief / reveals_intention / predicts_action`

**约束**：无环 + 时间单调；`predicts_action` 边连情感事件 → 未来行为事件，是 T7/T8 的 ground truth 提取依据。

输出：`data/event_graph/<series>/<episode>.json`

---

## Stage 4：集内情感轨迹 → M4 (per-episode)

对每集每个主角，沿事件 DAG 抽 emotion timeline：每个时间点 (emotion_label, intensity, trigger_event_id)。

---

## Stage 4.5：跨集 merge → M3 跨集事件图 + M4 累积轨迹

- **M3 构建**：把 10 集 M2 拼成全季 DAG，跨集边：人物连续性、剧情连续性、关系演化、long-term `causal`
- **M4 累积**：每个角色的情感轨迹连接成全季 timeline
- 输出 `data/season/<series>/M3.json`、`data/season/<series>/M4_<character>.json`

---

## Stage 4.6：语义记忆沉淀 → M5 / M6 / M7

用 **GPT-5（长上下文）** 一次吃完 M2-M4 全季内容，沉淀语义层：

| 输出 | 内容 |
|---|---|
| M5 人格档案 | OCEAN 五维数值 + 关键性格特质 + 行为模式 + 关键 backstory，含 evidence_refs (事件 id) |
| M6 关系网络 | 每对主角关系类型（family/love/rival/...）+ 关系演化曲线（哪集发生变化、变化方向）|
| M7 世界状态 | 剧情关键设定（"Walter 有癌症"）+ 设定时间戳 + 演化更新点 |

**与另两篇 Video Agent 协同**：M5/M6/M7 的 schema 直接复用 Video Agent 论文的 semantic memory schema，方便 E3 评估时把它们当 Agent 记忆系统输入。

---

## Stage 5：9 任务族问题生成

对每个 (event/relation/trajectory/profile) 触发点，按任务-M 依赖矩阵决定输入 context，调用对应任务的 prompt 模板。

**Prompt 设计原则**（9 个 prompt 文件 `prompts/qgen_T*.md`）：
- 输出 4 选 1 选择题（含 1 正确 + 3 distractor 占位，distractor 在 Stage 6.1 补全）
- 必带 `answer_evidence`: 时间戳 + 模态 + 涉及 M 组件清单
- T7/T8 强制要求 future ground truth 已观测（horizon 字段：1min/5min/30min/cross-episode）
- T9 的"个性"必须显式引用 M5 的 evidence_refs，杜绝幻觉

**配比目标**（参考 CharToM-QA 的情感主导 + 我们的预测主导）：
| 任务 | 占比 |
|---|:---:|
| T1 | 15% |
| T2 | 12% |
| T4 | 8% |
| T5 | 12% |
| T6 | 10% |
| T7 | 13% |
| T8 | 12% |
| T9 | 10% |
| T10 | 8% |

---

## Stage 6：6 道自动质量过滤

继承上版 6 道 filter，按任务族微调：

| Filter | 检测 | 处置 |
|---|---|---|
| F1 No-context | 仅 Q+4 选项给 GPT-5 → 正确率 >35% 即选项有偏 | 丢弃 |
| F2 Subtitle-only | 仅字幕能答对（仅对 T1/T2/T4 严格） | 命中→丢弃 |
| F3 Modal-ablation | mask 视觉/音频后正确率仍 >80% → 模态非必需 | 丢弃 |
| F4 Cross-model consensus | 3 strong MLLM (GPT-5 / Gemini 2.5 / Qwen2.5-Omni-72B) 一致率 <2/3 | 丢弃 |
| F5 Option-swap | 互换 correct/distractor 仍选原 correct → shortcut | 丢弃 |
| F6 Future-leak (仅 T7/T8) | 给完整 future 视频太易/太难 | 调整或丢弃 |

外加 **F7 M-dependency check**（新增）：
- 对每题，断言其声明的"必需 M 组件"是真必需——把它从 context 中扣除，正确率应显著下降；若没下降说明任务-M 依赖不真实，丢弃
- T8/T9/T10 必须通过 F7，是 long-term memory 题目真实性的最后防线

---

## Stage 7：评估面板

### 7.1 评估对象

| 类别 | 候选 |
|---|---|
| End-to-end MLLM | Qwen2.5-Omni-72B, GPT-5, Gemini 2.5 Pro, InternVL3, VideoLLaMA3 |
| Video Agent | M3-Agent（外部 baseline），本项目 Video Agent（论文 3），本项目 Long-Video Model（论文 2） |
| Text-only baseline | 仅给字幕的 GPT-5（无视频） |

### 7.2 评估 Setting 实现

四档 Setting 的语义、目标、成本对比详见前文「设计轴 A：Evaluation Setting (E0–E3)」。此处仅记 Stage 7 评估时的实现要点：

- E0：用 ffmpeg 按 `answer_evidence` 时间戳 ±2-3min 截取片段
- E1：整集 mp4 直接送
- E2：实现走 E2-summary 路线（每集摘要在 Stage 4.5 顺带产出，复用即可）；预算允许时对 GPT-5/Gemini 2.5 Pro 跑 E2-raw 作为参考点
- E3：M5/M6/M7 渲染为 Markdown 结构（JSON 仅 Agent 自动消费时用）

### 7.3 核心指标

- 9 任务各 task accuracy
- 跨 Setting gap：Acc(E1)-Acc(E0)、Acc(E2)-Acc(E1)、Acc(E3)-Acc(E2) → 量化各档增益
- Predictive horizon-aware accuracy：T7 (1/5min) × T8 (30min/cross-ep) 分档
- Multimodal evidence grounding F1（继承）
- Human-Model gap：人类组 A vs 最强模型

---

## Stage 8：人类对照实验（唯一人工环节）

200 题人类子集（中英各 100），16 名志愿者：
- 组 A（每语种 4 人）：看过全季全集 → 跑 E2/E3 同等信息量
- 组 B（每语种 4 人）：只看 2-5 min 片段 → 对标 E0
- 在 9 任务上跑 → 预期：
  - T1/T2 两组接近（局部就够）
  - T4/T5/T6/T7 A 组高 10-15pp
  - T8/T9/T10 A 组高 ≥20pp（长程记忆真正体现）

---

## 风险与缓解

| 风险 | 缓解 |
|---|---|
| 版权：视频不可分发 | 仅发 annotations + timestamps，用户自备视频（对齐 M3-Bench/MoMentS） |
| 全自动出题偏差 | 6+1 道 filter，特别 F7 保证 M 依赖真实 |
| M3 跨集 merge 错误 | 角色置信度 ≥0.8 才进 M3；M3 与字幕中"被称呼名字" 交叉验证 |
| M5/M6/M7 语义沉淀幻觉 | GPT-5 输出必带 evidence_refs，无 refs 字段一律丢；GPT-5-mini 二次校验 refs 是否真存在 |
| T8/T9 ground truth 偏差 | T8 future event 必须在 M3 中显式存在；T9 个性必须有 ≥3 evidence_refs |
| 中英维度不均 | 中英各独立出 QA，分层抽样到 final；评估报告分语种数 |
| Predictive horizon ground truth | 每条 T7/T8 题手动声明 horizon，pipeline 自动校验 future window 内事件存在 |

---

## 文件与目录结构

```
/Users/zhangshuo/ai_research/
├── benchmark/
│   ├── pipeline/
│   │   ├── stage1_perception.py
│   │   ├── stage2_events.py
│   │   ├── stage3_relations.py        # M2
│   │   ├── stage4_trajectory.py       # M4(per-ep)
│   │   ├── stage4_5_cross_episode.py  # M3 + M4 cumulative
│   │   ├── stage4_6_semantic.py       # M5/M6/M7
│   │   ├── stage5_qgen_T1..T10.py     # 9 任务出题
│   │   ├── stage6_filters.py          # F1-F7
│   │   └── stage7_eval.py
│   ├── prompts/
│   │   ├── event_extraction.md
│   │   ├── relation_extraction.md
│   │   ├── semantic_M5_persona.md
│   │   ├── semantic_M6_relation.md
│   │   ├── semantic_M7_world.md
│   │   ├── qgen_T1.md ... qgen_T10.md
│   │   └── judge_template.md
│   ├── data/
│   │   ├── raw/ perception/ events/ event_graph/
│   │   ├── trajectory/ season/ qa/ final/
│   └── README.md
└── (papers 2/3 ...)
```

---

## Verification 路线

1. **Pipeline 端到端 dry-run**：选 1 集 ~45min → Stage 1→6 跑通 → 期望 ~20-30 QA / 集
2. **3 集 pilot**：1 部剧 3 集 → 完成 Stage 4.5+4.6 跨集 merge → 期望 ~80-100 QA + M3-M7 文件可读
3. **小规模 baseline 验证**：用 ~250 题跑 GPT-5 / Qwen2.5-Omni / M3-Agent 在 E0-E3 → 验证 Setting gap 存在
4. **Sanity 人类对照**：2 人小组（1 看过 1 未看）30 题 → 验证 gap 方向正确
5. **F7 消融自检**：随机抽 50 道 T8/T9/T10，扣 M5/M6/M7 跑 Stage 5 重出题 → 应有显著质量下降

---

## 论文 Selling Points

1. **首个 9-任务 + 4-Setting 矩阵的情感长视频推理 benchmark**
2. **首次显式区分 M1-M7 7 个记忆组件** + 任务-记忆依赖矩阵 → 量化报告 benchmark 长程依赖密度
3. **新任务 T7/T8 情感→行动预测**：video-only 才能做，文本 ToM benchmark 没有
4. **新任务 T9 个性一致性（OCEAN × 长视频）**：依赖 M5 语义记忆，是 Video Agent 能力的核心评估题。**首个**从多集视频证据推断 OCEAN 并测人格一致性的 benchmark；现有 PsychoBench/Big5-Chat/CharacterEval 全 text-only，First Impressions V2 仅 15s clip，CPED 仅 utterance-level 中文 TV。
5. **新任务 T10 情感传染（关系图传播）**：依赖 M4×N + M6；**完全空白领域**——心理学有实证但无 AI benchmark。MELD/IEMOCAP/MERR 把情感视作 per-character 标签；MoMentS Emotions 维度不涉及传染；SIV-Bench 的 Relation Inference 仅是瓶颈而非图传播任务。我们强制 directionality。
6. **季级 cross-episode scale**：6 部剧 × 首季 10 集 = 60 集 ≈ 240-360 h，超过现有长视频 benchmark 最长设置（X-LeBench 16.4h life-log，VRBench 1.6h，LVBench 68min，HourVideo 120min）。
7. **中英双语并行**：现有中文长视频仅 Movie101（ACL 2023，前 MLLM 时代），中英协同评估首创。
8. **全自动 pipeline + F7 M-依赖过滤器**：方法论上扩展 HumanVBench（filter 流水线 + distractor-from-discarded-candidates）与 MoMentS（LLM-in-the-loop distractor evaluator），新增 F7 验证任务-记忆依赖真实性。
9. **协同评估**：与三篇论文的另两篇 Video Agent 在 M5/M6/M7 schema 上原生对齐；M3-Agent / GCAgent / WorldMM / Generative Agents 提供文本端 reference。E3 setting 首创 modular memory-injection 评测。

---

## Related Work：与现有 benchmark 的差异化定位

> 详见 `survey.md`。本节为论文 Related Work 写作时的核心对比矩阵。

### 与最接近的 7 个 benchmark 对比

| 与谁比 | 类型 | 共同点 | 我们的独占维度 |
|---|---|---|---|
| **CharToM-QA** (ACL 2025, arXiv 2501.01705) | 文本，1035 Q × 20 经典小说 | 4 心理维度（Belief/Intent/Emo/Desire）、长背景重要性、角色中心 | 多模态视频、跨集 continuity、OCEAN 显式测量、relational dynamics、post-cutoff 数据避免预训练污染 |
| **HumanVBench** (CVPR 2026, arXiv 2412.17574) | 视频，16 任务，4.7h Pexels | Filter pipeline、distractor 合成、emotion + face track | TV-集级 vs Pexels 短片、ToM/intent/persona 维度、relational/contagion 任务 |
| **MoMentS** (EMNLP-F 2025, arXiv 2507.04415) | 视频，168 短片 × 14min，ATOMS 7 | MCQ + LLM-in-loop distractor、视频+音频+文本 | 季级 TV vs 短片、character continuity 跨集、OCEAN trait inference、emotion contagion 图传播、9 任务覆盖更广 |
| **MTMEUR** (ACM MM 2025, arXiv 2508.16859) | 视频，1451 vid × <1min，多轮 emo+cause+future | 主题贴近（emo+cause+action） | clip <1min vs 季级、多 horizon（1/5/30min/cross-ep）、OCEAN/contagion 独占 |
| **MME-Emotion** (2508.09210) | 视频，6000+ clips，8 emotion 任务 | 8 emotion task + CoT scoring | clip 级 vs 季级、+T6/T7/T8/T9/T10 新任务、4-setting matrix |
| **SeriesBench** (CVPR 2025, arXiv 2504.21435) | 视频，105 剧 × 28 叙事任务 | TV 剧集叙事 + character | 28 narrative tasks vs 9 emotional/ToM tasks、显式 memory regime、E3 setting、跨集 memory injection |
| **VRBench** (ICCV 2025, arXiv 2506.10857) | 视频，1010 vid avg 1.6h，多步因果 | 多步推理 + 时间戳 reasoning steps | 1.6h vs 季级、generic narrative 而非情感、无 OCEAN/contagion |

### 与 Video Agent baselines 的对比

| 系统 | M1 | M2 | M3 | M4 | M5 | M6 | M7 |
|---|---|---|---|---|---|---|---|
| **M3-Agent** (arXiv 2508.09736) | ✓ | 部分 | ✗ | ✗ | 部分 (entity) | 部分 (id-link) | 部分 (semantic) |
| **WorldMM** (arXiv 2512.02425) | ✓ | 部分 | ✗ | ✗ | ✗ | ✗ | ✓ semantic |
| **GCAgent** (arXiv 2511.12027) | ✓ | **✓** causal+temporal | ✗ | ✗ | 部分 (role) | ✗ | 部分 (situation) |
| **VideoForest** (arXiv 2508.03039) | ✓ | ✗ | 部分 (person) | ✗ | ✗ | 部分 (traj) | ✗ |
| **LongVideoAgent** (arXiv 2512.20618) | ✓ | ✗ | 部分 (episode) | ✗ | ✗ | ✗ | ✗ |
| **MECD+ / GraphThinker** | ✓ | **✓** causal | ✗ | ✗ | ✗ | ✗ | ✗ |
| **MovieChat / MA-LMM** | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Generative Agents** (text-only ref) | N/A | N/A | N/A | 部分 (mood) | **✓** | 部分 (agent-agent) | 部分 (town) |

**关键结论**：**无任何系统同时覆盖 M3+M4+M5+M6+M7**。M3-Agent 与 GCAgent 在结构上最接近（entity graph + event graph）但都不抽人格 trait 或情感轨迹；Generative Agents 拥有正确 persona/memory 范式但是文本端。本 benchmark 是首个评估 M1-M7 完整记忆栈的工作。

---

## 方法学细节补充（基于 HumanVBench / MoMentS / TVBench 等借鉴）

### Stage 5 出题流程的方法学改进

**沿用 HumanVBench distractor pipeline（arXiv 2412.17574）**：
1. 按任务族选定视频片段 / 事件触发点
2. 生成 Q/A 时**显式标注**视频中的关键 face crop / 时间戳 / 事件 id
3. **3 个独立 MLLM 迭代优化答案**：GPT-5 + Qwen2.5-Omni-72B + Gemini 2.5 Pro 各生成候选答案 → 共识为 correct → 异议为 distractor 候选库
4. **Distractor 从被舍弃的候选答案中生成**（不是手工编造）—— 这是 HumanVBench 的方法学创新
5. **6% answer-leakage 过滤**：HumanVBench 标准 leak rate

**沿用 MoMentS LLM-in-the-loop distractor evaluator（arXiv 2507.04415）**：
- 每条 QA 写完后，过 LLM evaluator 实时打分 distractor set 的 bias（是否有过于明显的"对/错"模式）
- pilot 阶段验证该方法可减少 distractor bias 19-24pp
- 我们的 prompt 在 `prompts/distractor_evaluator.md`

### Stage 6 Filter 增强（基于 TVBench 等）

**新增 F0 TVBench 风格 shortcut 三诊断**（前置于 F1-F7，是更严格的 sanity check）：
- **F0a Single-frame solvable**：随机抽 1 帧问问题，正确率 >30% → 视觉不依赖 → 丢弃
- **F0b Text-only solvable**：仅给问题+选项（无任何视频/字幕）正确率 >30% → 文本捷径 → 丢弃（F1 是更严的 GPT-5 跑 35%）
- **F0c World-knowledge solvable**：给问题+选项+剧名+季集号（无视频）正确率 >30% → 世界知识可答 → 丢弃

**新增 F8 MCQ→Free-form 转换抽查**（基于 VideoEval-Pro arXiv 2505.14640）：
- 每 task 抽 10% 的题目转成自由生成版，用 GPT-judge 评分
- MCQ acc 与 free-form acc 差值 >30% → 选项偏置严重 → 丢弃该 batch
- 保留两种 evaluation：MCQ 与 free-form 各报一组数据

**沿用 InfiniBench 元数据泄露诊断**（arXiv 2406.19875）：
- 仅给 title + episode title（无视频）正确率应 ≤random — 否则丢弃

**沿用 MF² paired claim 设计**（arXiv 2506.06275）：
- T1/T5 任务考虑做 paired (fact, fib) claim 评测，消除选项位置 bias

### Stage 1-4 中的 paralinguistic 音频处理强化

**关键发现**（基于 survey 5.3 + Omni-Emotion arXiv 2501.09502）：
- 仅 Gemini 2.5/3 native audio、GPT-4o/5 Realtime、Qwen-Omni 系列、video-SALMONN-2、MiniCPM-o 处理 raw waveform
- 字幕-only 模型损失 paralinguistic 信号 → 在 IEMOCAP 上 10-20pp emotion accuracy gap

**实战**：
- Stage 1 副语言 track 用 **Qwen2.5-Omni** 显式问"语调/停顿/颤抖/音量/叹息/笑声"
- 对**仅有视觉+字幕的模型**，提供两条评测路线：
  - (i) 字幕-only baseline
  - (ii) 字幕 + **audio-tagged metadata**（"[Walter sighs heavily at 14:32]"、"[Skyler voice trembles]"）— 由 Qwen-Omni 在 perception 阶段产出
- 报告中显示两个数字以量化 audio gap

### Stage 4.6 语义沉淀的 evidence 严格化（基于 CharToM-QA + M3-Agent）

**强制结构**：M5/M6/M7 每条 assertion 必须带：
1. `evidence_refs`: 至少 3 条 event id（CharToM-QA 标准）
2. `temporal_span`: 起止 episode + 时间戳
3. `confidence`: GPT-5 自报 + GPT-5-mini 二次校验
4. `cross_check`: face_cluster_id / character_id 与字幕"被称呼名字"双重 link（M3-Agent 思路）

**OCEAN 抽取**特别注意：
- 每维 5 个 evidence_refs（PsychoBench 量表的最低门槛）
- 跨集 stability check：把第 1-5 集证据和第 6-10 集证据**独立**做 OCEAN inference，比对 cosine similarity ≥ 0.8 才算 stable persona

---

## 评测 baseline 详表（基于 survey 5.4 推荐）

### 完整 baseline matrix

| Tier | 模型 | 类型 | 优势 | 估算成本（1800 QA × 4 setting） |
|---|---|---|---|---|
| **T1 闭源** | Gemini 2.5 Pro | end-to-end MLLM | 1-2M token, native audio, 6h video | $200-400 API |
| | Gemini 3 Pro | end-to-end MLLM | 10-FPS thinking mode, AV streaming | $300-500 API |
| | GPT-5 / 5.5 | end-to-end MLLM | 强 reasoning + Realtime audio | $600-1200 API |
| **T2 开源 AV** | Qwen3-Omni-30B-A3B | end-to-end MLLM | native waveform Apache 2.0, 22/36 AV SOTA | $300-600 GPU |
| | Qwen2.5-Omni-7B | end-to-end MLLM | Thinker-Talker, V-MME 64.3/72.4 | $150-300 GPU |
| | video-SALMONN 2-72B | end-to-end MLLM | 学术 AV 顶配, 72B 超 GPT-4o on AV-aware | $800-1500 GPU |
| **T3 开源 VL** | Qwen3-VL-235B-A22B-Thinking | end-to-end MLLM | 2h native, 顶级开源 VL | $1500-3000 GPU |
| | InternVL3.5-78B | end-to-end MLLM | V2PE 长 context | $1000-1800 GPU |
| | Qwen2.5-VL-72B | end-to-end MLLM | Multi-hour, sec-level localization | $700-1200 GPU |
| **T4 长视频专家** | VideoChat-Flash-7B | end-to-end MLLM | 16 tok/frame, 10K-frame NIAH 99.1% | $100-200 GPU |
| | LongVU-7B | end-to-end MLLM | hour-scale efficient | $100-200 GPU |
| | Apollo-7B (Meta) | end-to-end MLLM | hour-scale, MLVU 70.9% | $100-200 GPU |
| **Video Agent** | **M3-Agent** | agent | entity-centric memory graph, KEY baseline | $400-800 GPU |
| | GCAgent | agent | schematic + narrative episodic | $300-600 GPU |
| | WorldMM | agent | 3-memory adaptive retrieval | $300-600 GPU |
| | Deep Video Discovery | agent | LVBench 74.2% leader | $400-800 GPU |
| | VideoDeepResearch | agent | LRM + multimodal toolkit | $300-600 GPU |
| | VideoAgent (Wang/Fan) | agent | ECCV'24 reference | $200-400 GPU |
| | DoraemonGPT / TraveLER / OmAgent / VideoTree / DrVideo / Vgent | agent | 多种 reasoning paradigm | $100-200 GPU each (select 2-3) |
| **Cross-episode 专用** | LongVideoAgent | agent | multi-agent multi-episode reward-driven | $300-600 GPU |
| | VideoForest | agent | person-anchored cross-video | $200-400 GPU |
| **Emotion-specialized** | Emotion-LLaMA / v2 | emotion MLLM | MER2023 F1 0.9036, DFEW 45.59/59.37 | $200-400 GPU |
| | AffectGPT | emotion MLLM | ICML'25 Oral, 115K samples 2K labels | $200-400 GPU |
| | R1-Omni | emotion MLLM | RLVR reasoning | $200-400 GPU |
| | AVERE / EmoReAlM | emotion MLLM | AVEm-DPO, +6-19% | $200-400 GPU |
| | VidEmo | emotion MLLM | affective-tree, 69.3% | $300-600 GPU |
| **Text-only 消融** | GPT-5 字幕-only | text baseline | 验证视觉必要性 | $100-200 API |
| | Qwen3-32B 字幕-only | text baseline | 同上 | $50-100 GPU |
| | Generative Agents (字幕+提取 event caption) | text agent | persona memory reference | $100-200 API |

**总预算估算**：
- 完整 sweep（~30 models × 4 settings × 1800 QA）：~$10-15K
- 推荐主 slate（10 model：Gemini 2.5 Pro + GPT-5 + Qwen3-Omni + Qwen3-VL + video-SALMONN-2-72B + M3-Agent + GCAgent + VideoChat-Flash + Emotion-LLaMA + GPT-5 字幕-only）：~$5-8K
- 最小可行 5 model slate：$1500-2500

### 评测维度
按 `survey.md` 6.1 提出的 metric 框架，每个模型在每个 setting 上报告：
- 9 任务 task accuracy（MCQ）
- 抽样 free-form accuracy（GPT-judge）
- 跨 setting gap：Acc(E1)-Acc(E0)、Acc(E2)-Acc(E1)、Acc(E3)-Acc(E2)
- Predictive horizon-aware accuracy：T7 (1/5min) × T8 (30min/cross-ep)
- Multimodal evidence grounding F1
- OCEAN MAE（T9）
- Contagion-consistency score（T10）：方向准确率 + 强度 MAE
- Human-Model gap

---

## 预训练污染缓解（基于 CharToM-QA 警告）

CharToM-QA 显式承认经典小说在预训练 corpus 中深度暴露是其 confound。我们的缓解：

1. **首选 post-cutoff TV 剧（2024-2025 首播 + 部分 2022-2023）**：模型预训练 cutoff 普遍 2023 末或 2024。建议中英各选 1 部 2024-2025 首播剧 + 1 部 2023 剧 + 1 部更早经典剧作为对照。
2. **报告"剧名识别 baseline"**：用 GPT-5 仅看剧名问 plot summary → 若正确率高说明剧本身在预训练里有 metadata
3. **subset by knowledge level**：把 60 集分两子集：（a）模型完全没见过的 post-cutoff、（b）已经在网上有充足 wiki 的，分开报性能
4. **T9 OCEAN 任务对照**：让模型 vs 字幕-only 两种条件，gap = 真正的视频贡献
5. **避免用诸如 *Friends* 等过度训练的 TV 剧**（与 TVQA/MELD 重叠）

---

## 副语言 / Audio-tagged Evaluation 子集（新设计）

针对仅有视觉+字幕的强 MLLM（Qwen3-VL、InternVL3.5、Claude Opus 等不接 audio），单独设计**audio-tagged metadata** subset：

**抽取**：Qwen2.5-Omni 在 Stage 1 perception 阶段，把副语言信号显式渲染成文本标签
- 时间戳粒度：utterance 级
- 标签类别：`[语调: 颤抖/平淡/激动/讽刺]`、`[停顿: 短促/拖长]`、`[非语言: 叹气/笑/哭/咳嗽]`、`[音量: 低声/喊叫]`

**评测**：对每个 audio-blind 模型报告两组数字：
- (i) 纯字幕 baseline
- (ii) 字幕 + audio-tagged metadata 注入

**预期**：(ii) - (i) gap 即为 paralinguistic 信号的可恢复部分。Gemini/Qwen-Omni 等 native audio 模型作为上限对照。

这个设计参考 Omni-Emotion（arXiv 2501.09502）发现 audio-rich 模型在 IEMOCAP 上 +10-20pp。

---

## Verification 路线（补充）

继 1-5 项基础 verification 后增加：

6. **3 个直接竞品对照运行**：在我们 250 题 pilot 上跑 CharToM-QA、HumanVBench、MoMentS 的官方 MCQ 评测脚本 → 验证我们的 distractor 难度与他们 comparable（避免太简单或太难）
7. **F0a/F0b/F0c shortcut 三诊断**：单帧 / text-only / world-knowledge 三种 ablation 各跑一遍 → 应 ≤30% 正确率
8. **MCQ→free-form 转换抽查**（F8）：随机抽 10% 转 free-form，gap 应 ≤30%
9. **Audio-tag subset 验证**：在 60 题 audio-rich 子集上比较 (i) vs (ii)，应有 5pp+ gap
10. **预训练污染检查**：仅给剧名跑 GPT-5 plot summary → 验证 post-cutoff 剧选择有效
