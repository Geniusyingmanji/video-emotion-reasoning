# Survey：情感长视频推理 Benchmark 相关工作综述

> **目的**：为本项目「情感长视频推理 Benchmark」的设计建立完整的领域知识地图。涵盖（1）长视频理解 benchmark；（2）多模态情感分析 benchmark；（3）Theory-of-Mind 与角色/叙事推理；（4）Video Agent 与长视频记忆系统；（5）视频 MLLM 模型生态；并通过对最直接竞品的深度对比，定位本项目的差异化贡献。
>
> **调研截止**：2026-05；优先 2024-2026 工作；最相关的早期工作（IEMOCAP/MELD/TVQA/MovieGraphs 等）保留为基线参考。

---

## 0. TL;DR — 核心发现与定位

### 0.1 我们与现有 benchmark 的差异化
本项目同时占据**四个空白区**——这是任何已有 benchmark 都未覆盖的：

1. **季级时长**（10 集 × 45 min ≈ 7-8 h，最长达 60 集），现有"长视频" benchmark 大多 ≤1 h（Video-MME long 60 min；LongVideoBench 1 h；LVBench 68 min；HourVideo 120 min；VRBench 1.6 h），最接近的多剧集工作 LongTVQA/LongTVQA+（2512.20618）和 SeriesBench（2504.21435）也尚未做我们的「跨集情感推理 × OCEAN × 关系图传染」组合。

2. **跨集情感连续性**（M3+M4），现有情感 benchmark（MELD/M3ED/MEmoR/MME-Emotion/MTMEUR/HumanVBench/MoMentS）全部仅在 utterance / clip / 短片层级评测情感，**没有跨集情感轨迹**。

3. **OCEAN 人格 × 视频推理**（T9，依赖 M5），唯一具有 OCEAN 标签的视频 benchmark 是 **First Impressions V2**（15 秒 YouTube 自介绍）和 **CPED**（utterance-level 中文 TV），都在短片/单语级。没有任何 benchmark 从多集视频证据中推断 OCEAN 并测试人格在不同情境下的一致性。

4. **情感传染 / 人际情感影响**（T10，依赖 M6），心理学有实证研究但 **AI benchmark 完全不存在**。MELD/IEMOCAP/MERR 把情感当作每个角色独立的标签；MoMentS 的 Emotions 是个体维度；SIV-Bench 的 Relation Inference 仅是瓶颈而非图结构传播任务。

### 0.2 最直接的三个竞品 / 灵感来源

| 竞品 | 类型 | 与本项目关系 |
|---|---|---|
| **HumanVBench**（arXiv 2412.17574, CVPR'26） | 视频，16 任务，4.7h Pexels | **方法学起点** — distractor-from-discarded-candidates pipeline、3-MLLM iterative answer optimization、6% answer-leakage 过滤。复用。 |
| **MoMentS**（arXiv 2507.04415, EMNLP'25-F） | 视频，ATOMS 7 任务，168 短片（avg 14 min） | **最直接 ToM 视频对手**。复用 LLM-in-the-loop distractor evaluator（reduces bias 19-24pp）。我们在长度/跨集/OCEAN/传染上扩展。 |
| **CharToM-QA**（arXiv 2501.01705, ACL'25） | 文本，1035 Q × 20 经典小说，Belief/Intent/Emo/Desire | **最直接文本 ToM 对手**。读者笔记 → 4 维 ToM。人类 familiar 59-62% vs unfamiliar 41-49%（21pp gap 来自长 context）。我们对应到视频长程上下文。 |

其他需要在 paper 中明确对比的工作：**MME-Emotion**（2508.09210，最大 MLLM 情感 benchmark，clip 级）、**MTMEUR**（2508.16859，单论多轮情感+原因+未来行动；clip <1min）、**SeriesBench**（2504.21435，105 部 105 集叙事任务但不聚焦情感）、**VRBench**（2506.10857，多步因果但 1.6h 不跨集且非情感）、**M3-Agent**（2508.09736，最接近的 video agent 记忆系统）。

### 0.3 推荐评测模型（结合 cost & coverage）

**Tier 1 闭源**：Gemini 2.5/3 Pro（原生音频 prosody + 1-2M token 全季 ingest）、GPT-5/5.5（强 reasoning + Realtime audio）

**Tier 2 开源音视频统一**：Qwen3-Omni-30B-A3B（开源 AV 顶配；native waveform）、video-SALMONN 2-72B（72B 在 AV-aware V-MME 超 GPT-4o）

**Tier 3 开源 VL 前沿**：Qwen3-VL-235B、InternVL3.5-78B、VideoChat-Flash-7B（efficient 3h 长视频）

**Video Agent baselines**：M3-Agent（KEY）、GCAgent、WorldMM、Deep Video Discovery、VideoDeepResearch、VideoExplorer、VideoAgent (Wang/Fan)、TraveLER、OmAgent、VideoTree、DrVideo、Vgent；**+ Generative Agents** 作为文本端记忆 reference。

预估全套 7 模型 × 1800 QA × 4 settings sweep：**$4500-8700**（闭源 API $800-1600 + 开源 GPU $3700-7100）。最小可行 5 模型 slate：$1500-2500。

---

## 1. 长视频理解 Benchmark

### 1.1 总体格局
2023 年以来出现三个演化轴：
- **时长**：分钟 → 小时（HourVideo 1-2h、X-LeBench 16.4h）
- **任务复杂度**：感知描述 → 多步因果时序推理
- **标注严格度**：纯人工 → LLM+人混合 → 严格 shortcut/leakage 过滤

但**领先 benchmark 几乎都是从长视频里抽短片段问问题**——MLVU 平均 12 min、Video-MME 平均 17 min、CinePile 2 min/clip。前沿模型 vs 人类在小时级 benchmark 上差距巨大：HourVideo Gemini-1.5-Pro 37.3% vs 人类 85.0%；MME-Emotion 39.3% recognition / 56.0% CoT。

VideoEval-Pro（2505.14640）发现把 MCQ 改成 free-form 后性能掉 25%+，说明很多"长视频" benchmark 的高分实际是选项偏置。

### 1.2 关键 benchmark 简表

| 类型 | Benchmark | 年份/会议 | arXiv | 时长 / 规模 | 主要任务 | 备注 |
|---|---|---|---|---|---|---|
| **综合长视频** | Video-MME | CVPR 2025 | 2405.21075 | 11s-1h, 254h, 900 vid / 2700 QA | 12 任务 short/medium/long | 工业标准；Gemini 2.5 Pro 84.8% |
| | MLVU | CVPR 2025 | 2406.04264 | 3min-2h avg 12m | 9 任务 incl PQA / NeedleQA | GPT-4o 64.6% |
| | LongVideoBench | NeurIPS 2024 D&B | 2407.15754 | up to 1h, 3763 vid / 6678 QA | 17 类 referring-reasoning | GPT-4o 66% / XL 50.5% |
| | LVBench | ICCV 2025 | 2406.08035 | avg 68min, 117h, 103 vid / 1549 QA | 6 类 21 子任务 | TV+sports+CCTV |
| | InfiniBench | EMNLP 2025 | 2406.19875 | avg 53min, >1000h, 87.7K QA | 8 skills + "Movie Spoiler" | GPT-4o 49.16% MCQ |
| | CG-Bench | ICLR 2025 | 2412.12075 | 1219 vid / 12129 QA | Percep/reason/hallu + clue-ground | Open-source gap |
| | MovieChat-1K | CVPR 2024 | 2307.16449 | 1K movies/TV 15 genres | Global vs breakpoint | Memory bank |
| | MoVQA | 2023 | 2312.04817 | 7.5/20/120min × 100 movies | 6 question types | Length-grouped |
| | CinePile | NeurIPS 2024 D&B | 2405.08813 | 9396 movie clips ~2min | 9-axis 305K MCQ | Audio Desc + LLM |
| | MF² | 2025 | 2506.06275 | 50-170min × 50+ movies | Paired fact/fib motiv+emo+causal | Gemini-2.5-Pro short |
| | MovieCORE | EMNLP 2025 Oral | 2508.19026 | Movies | Bloom 4.9 System-2 cognitive QA | ACE +25% |
| **TV/叙事** | TVQA/TVQA+ | EMNLP 2018/ACL 2020 | 1809.01696 | 460h, 152K QA, 6 TV shows | Compositional dialogue+visual | Pre-MLLM; Goldfish-long 41.78% |
| | DramaQA | AAAI 2021 | 2005.03356 | Korean drama | 4-level cognitive hier | 17983 QA |
| | **SeriesBench** | CVPR 2025 | 2504.21435 | 105 drama series, 1072 vid | 28 narrative tasks 5 cat | PC-DCoT +10% |
| | **VRBench** | ICCV 2025 | 2506.10857 | 1010 narrative vid avg 1.6h | 9468 QA + 30292 reasoning steps | 7 reasoning types incl event attribution |
| | LongTVQA/+ | 2025 | 2512.20618 | Multi-episode TV | Episode-level reward-driven RL | LongVideoAgent multi-agent |
| **Egocentric 长** | Ego4D | CVPR 2022 | 2110.07058 | 3670h egocentric | Episodic mem / hand-obj / AV / forecast | 931 wearers |
| | EgoSchema | NeurIPS 2023 D&B | 2308.09126 | 3min/250h, 5031 Q | Long-form QA + temporal certificate | GPT-4.1 76.1% |
| | HourVideo | NeurIPS 2024 D&B | 2411.04998 | 20-120min, 381h, 500 vid / 12976 Q | 18 sub: summ/percep/reason/nav | Gemini-1.5-Pro 37.3% vs human 85% |
| | X-LeBench | EMNLP-F 2025 | 2501.06835 | 23min-16.4h life-logs | Temporal loc/summ/count/order | All baselines weak |
| **时序推理** | MVBench | CVPR 2024 | 2311.17005 | <1min × 11 datasets | 20 temporal tasks static→dynamic | VideoChat2 51% |
| | TempCompass | ACL-F 2024 | 2403.00476 | Conflicting videos short | 4 task types | -- |
| | TVBench | BMVC 2025 | 2410.07752 | Short conflicting | Pure temporal w/ shortcut filters | Diagnostic |
| | E.T. Bench | NeurIPS 2024 D&B | 2409.18111 | 7K vid 251h | 12 tasks event-level | -- |
| **因果/反事实** | NExT-QA | CVPR 2021 | 2105.08276 | 5440 daily | Causal 48%/temp 29%/desc 23% | 52K QA |
| | NExT-GQA | CVPR 2024 H | 2309.01327 | NExT-QA + grounding | 10.5K 时序 grounding label | Forces visual grounding |
| | CausalVidQA | CVPR 2022 | -- | Movies/daily | Desc/explain/predict/counterfactual | Pre-MLLM |
| | CausalVQA (Meta) | 2025 | 2506.09943 | Physical short | 793 paired Q counterfactual/anticip/plan | Gemini-2.5-Flash best |
| | Perception Test | NeurIPS 2023 D&B | 2305.13786 | 11.6K real 23s | Memory/Abstract/Physics/Semantic | Human 91.4% vs SOTA 46.2% |
| **诊断/合成** | VideoNIAH/VNBench | 2024 | 2406.09367 | Synth needle 10-30min | Retrieval/order/count | Long-distance dependency |
| | VideoEval-Pro | 2025 | 2505.14640 | Agg avg 38min | Free-form versions | Drops >25% vs MCQ |
| **中文** | Movie101 | ACL 2023 | 2305.12140 | 中文电影 92h / 30174 clips | MCN narration + TNG | Pre-MLLM; 唯一大规模中文 |

### 1.3 长视频 benchmark 中的 quality 控制方法
我们在 Stage 6 的 6+1 filter 设计可以从以下工作中借鉴：
- **TVBench**：3 大 shortcut 诊断（single-frame solvable / text-only solvable / world-knowledge solvable）
- **MF²**：成对正反 claim 评测消除选项位置偏置
- **CG-Bench**：clue-grounded white-box & black-box 双重评测
- **InfiniBench**：title-only 评测度量 metadata 泄露
- **VideoEval-Pro**：MCQ→free-form 转换量化选项偏置
- **MoMentS**：LLM-in-the-loop distractor evaluator（减少 19-24pp bias）
- **HumanVBench**：多 MLLM 迭代答案 + 6% answer-leakage 过滤

### 1.4 与本项目相关的缺口
1. **季级跨集**：无现存 benchmark 跨完整 TV 季节，LongTVQA/+ 与 SeriesBench 是仅有的近似
2. **跨集情感**：MME-Emotion clip 级、MTMEUR <1min；无 benchmark 在多集 narrative 上测情感
3. **个性一致性 × 长视频**：SeriesBench 有 key character portraits 但无 formal consistency eval
4. **情感传染**：完全空白
5. **分层 memory regime（E0/E1/E2/E3）**：无 benchmark 显式变 input scope 同时固定 question
6. **中文长视频推理**：Movie101 唯一但前 MLLM 时代
7. **Agent vs end-to-end MLLM 公平对比**：当前 apples-to-oranges
8. **情感证据 multimodal grounding**：NExT-GQA/CG-Bench 强 grounding 但非情感线索
9. **短时 vs 跨集行动预测**：无 benchmark 分层度量
10. **情感转折定位**：TVBench/E.T. Bench 时序 grounding 但非情感转折

---

## 2. 多模态情感分析 Benchmark

### 2.1 三个时代
- **经典 ERC（2008-2022）**：IEMOCAP、MELD、EmoryNLP、MOSEI（utterance 级，固定标签，F1/UAR/WAR/MAE）
- **In-the-wild 面部（2020-2023）**：DFEW、FERV39k、MAFW、Aff-Wild2/ABAW（万级 clip）
- **MLLM 时代（2024-2026）**：EmoBench、EmoBench-M、MME-Emotion、MTMEUR、AffectGPT、Emotion-LLaMA、R1-Omni、AVERE、VidEmo（推理链、开放词表、显式因果）

### 2.2 关键 benchmark 简表

| 类型 | Benchmark | 年份 | arXiv | 模态/来源 | 标签 | 规模 | SOTA |
|---|---|---|---|---|---|---|---|
| **经典 ERC** | IEMOCAP | 2008 LREC | -- | AVT+MoCap / 实验室 dyad | 5-6 basic + VAD | 12h, 7311 utt | F1 ~80% |
| | MELD | ACL 2019 | 1810.02508 | AVT / Friends | 7 Ekman + sentiment | 1433/13708 utt | F1 ~67% |
| | EmoryNLP | AAAI-W 2018 | -- | T (后 AV) / Friends | 7 affect | 897/12606 utt | F1 40-42% |
| | CMU-MOSEI | ACL 2018 | -- | AVT / YouTube | 6 Ekman + VAD | 23.5K utt | F1 ~50% |
| | M3ED | ACL 2022 | 2205.10237 | AVT / 56 中文 TV | 7 emo 中文 | 990/24449 utt | F1 50-55% |
| | MEmoR | ACM MM 2020 | -- | AVT / BBT | 14 fine | 5502/8536 samples | Acc ~50% |
| | **CPED** | 2022 | 2205.14727 | AVT / 40 中文 TV | 13 emo + **OCEAN** + 19 DA | 12K/133K, 392 speakers | 唯一含 OCEAN 的 ERC |
| **In-the-wild 面部** | DFEW | ACM MM 2020 | 2008.05924 | V / 1500+ 电影 | 7 basic | 16372 clips | UAR 50-60% |
| | FERV39k | CVPR 2022 | 2203.09463 | V / 4K YouTube | 7 basic | 38935 clips | UAR 40-45% |
| | MAFW | ACM MM 2022 | 2208.00847 | AVT / Movies+TV+短 | 11 single + 32 compound 双语 | 10045 clips | UAR 40% / WAR 50% |
| | Aff-Wild2 / ABAW | 年度挑战 | 2106.15318 | AV / YouTube wild | 7 + VAD + 12 AUs | 564 vid / 2.8M frame | CCC ~0.55 VA |
| | CAER/CAER-S | ICCV 2019 | 1908.05913 | V / 79 TV | 7 basic | 13201 vid / 70K static | Acc ~80% |
| | **First Impressions V2** | ECCV-W 2016 | -- | AVT / YouTube 15s | **OCEAN** regression | ~10K clips | MAE 0.10-0.12 |
| **情感因果** | ECE/ECPE | ACL 2019 Outstanding | 1906.01267 | T / 中文新闻 | Pair extraction | NTCIR-13 ~2K docs | F1 ~62-72% |
| | RECCON | 2021 | -- | T / IEMOCAP+DailyDialog | CSE + CEE | 1106 dialogs / >10K pairs | F1 ~70-75% |
| | MECPE/SemEval-2024 T3 | 2024 | 2405.13049 | AVT / MELD | Multimodal emo-cause pair | ~13K utt | Weighted F1 0.34（winning） |
| | EIBench (Why-We-Feel) | CVPR-W 2025 | 2504.07521 | VT / 多源 | Free-form emo + cause | 1615 basic + 50 complex | Gap on complex |
| **LLM 情感 IQ** | EmoBench | ACL 2024 | 2402.12071 | T / 手工 | EI EN+ZH | 400 Q | GPT-4 << human |
| | **EmoBench-M** | 2025 Feb | 2502.04424 | AVT / 实验室+社交 | 13 EI scenarios × 3 dims | Multi-clip | Gemini-3.0-Pro 70.5 / GPT-5.2 66.5 |
| | EmoLLM+EmoBench (Yang) | 2024 | 2406.16442 | V/Video/T | 5 emotion tasks 287K instr | 287K pairs | +12.1% |
| | **MME-Emotion** | 2025 Aug | 2508.09210 | AVT | 8 任务: ER-Lab/Wild/Noise, FG-ER, ML-ER, SA, FG-SA, IR | 6000+ clips | 39.3% recog / 56% CoT |
| | **MTMEUR** | ACM MM 2025 | 2508.16859 | AVT / 真实场景 | 多轮 emo+cause+future action | 1451 vid / 5101 Q | 29.10%-71.19% |
| | EMER | 2023 | 2306.15401 | AVT | Explainable + OV MER | 332 samples | -- |
| | OV-MER/OV-MERD | 2024 | 2410.01495 | AVT | 248-term open vocab | 332/avg 3.34 labels | -- |
| | AffectGPT / MERR | ICML 2025 Oral | 2501.16566 | AVT / web | 2K fine emo / 115K samples | 115K | New SOTA |
| | **Emotion-LLaMA** | NeurIPS 2024 | 2406.11161 | AVT | HuBERT+MAE+EVA-CLIP → LLaMA + MERR | 28618 coarse + 4487 fine | F1 0.9036 MER2023; DFEW 45.59/59.37 |
| | R1-Omni | 2025 Mar | 2503.05379 | AVT | RLVR-based reasoning | DFEW/MAFW/RAVDESS + cold-start | OOD ↑ |
| | AVERE / EmoReAlM | ICLR 2026 | 2602.07054 | AVT | Pref-opt DFEW+EMER+RAVDESS | -- | +6-19% over baselines |
| | CA-MER | ACM MM 2025 | 2508.01181 | AVT | Modality conflict 3 subsets | -- | MoSEAR closes gap |
| | **VidEmo** | 2025 Nov | 2511.02712 | AVT / curated | Affective-tree reasoning Emo-CFG 2.1M instr | -- | 69.3% avg |
| | MERBench | 2024 | 2401.03429 | AVT | Unified MER eval | Standard | Reproducible |
| **角色/人格/ToM** | DialogRE | ACL 2020 | 2004.08056 | T / Friends | 36 relations | 1788/8119 pairs | F1 ~50-60% |
| | PsychoBench | 2024 | -- | T | 11 inventory incl OCEAN | 13 scales | OCEAN-stable partly |
| | CharacterEval | 2024 ACL | 2401.01275 | T 中文 | Role-play 13 metrics × 4 dim | 11376 ex / 77 chars | CharacterGLM strong |
| | **CharToM-QA** | 2025 ACL | 2501.01705 | T / classic novels | Char ToM 4 dim | 1035 Q / 20 books | DS-R1 54.0%, GPT-4o 51.8-58.7% |
| | **EmoTx** | CVPR 2023 | 2304.05634 | AVT / MovieGraphs | Multi-label emo @ scene+char | 51 movies 26-181 labels | mAP gains |
| | MovieGraphs | CVPR 2018 | 1712.06761 | V+graph / 51 movies | Char attr/relation/interaction/situation/reason | 7637 clip graphs | -- |

### 2.3 主流方法家族

5 个 SOTA 方法系：
1. **Encoder-fusion + instruction-tune（Emotion-LLaMA 家族）**：HuBERT+MAE+EVA-CLIP→LLaMA via MLP, MERR 训练。MER2023 F1 0.9036、DFEW 45.59/59.37。AffectGPT 加 pre-fusion；EmoLLM 加 multi-perspective + EmoPrompt。
2. **RL-based reasoning（R1-Omni / AVERE）**：RLVR+GRPO 或 DPO（AVEm-DPO 引入 hallucination 偏好和 text-prior penalty）。zero-shot DFEW/RAVDESS/EMER +6-19%。
3. **Affective-tree / CoT（VidEmo / EIBench）**：3-level hierarchical, 2-stage 调优。69.3% avg。
4. **Conflict-aware（CA-MER / MoSEAR）**：SOTA 系统性 over-rely on audio。MoSEAR 用 parameter-efficient modality 专家 + attention reallocation 修复。
5. **Audio-prosody omni（Qwen2.5/3-Omni）**：TMRoPE + Thinker-Talker，原生 waveform 处理 prosody/tone/emotion。

经典 fusion 仍是基线：TFN（1707.07250）、MulT、MISA、Self-MM、MMGCN、DialogueRNN、DAG-ERC、EmoTx、EmoShiftNet。

### 2.4 与本项目相关的缺口

| # | 缺口 | 现状 | 本项目 |
|---|---|---|---|
| 1 | 跨集情感连续性 | MELD/M3ED utterance；MME-Emotion clip；MTMEUR <1min | T1+T4 全季 |
| 2 | 多跳情感因果链 | 所有 emotion-cause（ECPE/RECCON/MECPE/T3/EIBench）都是 single-hop | T6 multi-hop |
| 3 | 长上下文单跳情感因果 | RECCON/T3 是 short clip | T5 over 集级 |
| 4 | 情感证据 multimodal grounding | 通用 grounding（Open-o3 Video/VideoChat-R1）非情感 | T2 grounding 到 timestamp+face+dialog span |
| 5 | 短时 emo-driven action prediction | MTMEUR 有一个子任务但非 primary | T7 primary |
| 6 | 跨集 action prediction conditioned on emotion | 无 | T8 全新 |
| 7 | OCEAN × 长视频 | First Impressions V2 15s / CPED utt / CharacterEval text | T9 全新（首个 video OCEAN benchmark） |
| 8 | 情感传染 / 人际传染 | 心理学有，AI benchmark 无 | T10 全新 |
| 9 | 长上下文 modality conflict 和被掩盖情感 | CA-MER 揭示，无长视频 benchmark | 本 benchmark 自然涌现 |
| 10 | 长视频情感推理链的可复现 metric | 各 benchmark 用不同 metric 拼接 | 提出统一 9-metric 框架 |

---

## 3. Theory of Mind / 角色与叙事推理

### 3.1 总体景观

**文本 ToM** 蓬勃发展：ToMi（2019）→ SocialIQA → FANToM → HiToM → OpenToM → ToMBench → BigToM → SimpleToM → NegotiationToM → ExploreToM → DynToM → PersuasiveToM → **CharToM-QA**（2025 ACL，**最直接文本竞品**）→ CogToM。前沿 LLM 落后人类 10-44pp，gap 在 asymmetric / higher-order / applied 时扩大。

**视频 ToM** 稀疏且不成熟。完整清单：
- Social-IQ / Social-IQ 2.0（generic 社会智能）
- MMToM-QA（ACL'24 Outstanding，household synth）
- MuMA-ToM（AAAI'25 Oral，multi-agent）
- EgoToM（egocentric Ego4D）
- Through-the-ToM's-Eye / VToM
- **HumanVBench**（CVPR'26，方法学起点）
- **MoMentS**（EMNLP'25-F，最直接视频 ToM 对手）
- SIV-Bench（TikTok/YouTube）
- Social Genome（grounded reasoning traces）
- MovieCORE（System-2 cognitive QA）
- SeriesBench（叙事剧 28 任务）
- Mind the Motions / Motion2Mind（body language ToM）

### 3.2 关键 benchmark 简表

| 类型 | Benchmark | 年份 | arXiv | 任务 | Best |
|---|---|---|---|---|---|
| **文本 ToM** | ToMi | EMNLP 2019 | -- | 1st/2nd false belief | GPT-4 saturated ~80% |
| | SocialIQA | EMNLP 2019 | 1904.09728 | 9 ATOMIC dims | ~78% |
| | FANToM | EMNLP 2023 | 2310.15421 | Conv info asymmetry | Far below human |
| | HiToM | EMNLP-F 2023 | 2310.16755 | 1-4 阶 ToM | Sharp drop |
| | OpenToM | ACL 2024 | 2402.06044 | Phys/psych + personality 696 narratives 16008 Q | -- |
| | ToMBench | ACL 2024 | 2402.15052 | 8 tasks × 31 ability 双语 | GPT-4 -10% vs human |
| | BigToM | NeurIPS 2024 | -- | belief↔percept↔action graph | -- |
| | SimpleToM | 2024 | 2410.13648 | Mental + behavior + judgment | -- |
| | NegotiationToM | EMNLP-F 2024 | 2404.13627 | BDI in negotiation | Below human |
| | ExploreToM | 2024 | 2412.12175 | Adversarial A*+DSL | Llama-3.1-70B 0%, GPT-4o 9% |
| | DynToM | 2025 | 2505.17663 | Temporal evolution mental states | LLM avg 44.7% below |
| | **CharToM-QA** | ACL 2025 | 2501.01705 | Char ToM 经典小说 1035 Q | DS-R1 54%, GPT-4o 51.8-58.7%; human 59-62%/41-49% |
| **视频 ToM** | Social-IQ | CVPR 2019 | -- | Social QA | Top 64%, human 95% |
| | Social-IQ 2.0 | ICCV-W 2023 | -- | Multi-MCQ | LLaVA-Video ~70% |
| | MMToM-QA | ACL 2024 Outstanding | 2401.08743 | Belief+goal household | BIP-ALM > GPT-4 |
| | MuMA-ToM | AAAI 2025 Oral | 2408.12574 | Multi-agent | LIMP 76.6%, Gemini-1.5-Pro 56.4%, human 93.5% |
| | **HumanVBench** | CVPR 2026 (2024) | 2412.17574 | 16 tasks inner+outer | Gemini-1.5-Pro 65.5%, GPT-4o 51.7%, human 89.1% |
| | **MoMentS** | EMNLP-F 2025 | 2507.04415 | ATOMS 7 abilities 168 短片 | LLaVA-Video-72B 67.66%, human 86% |
| | EgoToM | 2025 | 2503.22152 | Goal/belief/action over Ego4D | MLLMs near-human goal |
| | SIV-Bench | 2025 Jun | 2506.05425 | SSU+SSR+SDP TikTok | Relation Inference bottleneck |
| | Social Genome | EMNLP 2025 | 2502.15109 | Grounded reasoning traces 272 vid | Gemini 74.4%, GPT-4o 71% |
| | Mind the Motions | 2025 Nov | 2511.15887 | 222 nonverbal cues × 397 mind states | Gap detection |
| **叙事/角色视频** | MovieGraphs | CVPR 2018 | 1712.06761 | 51 movies 7637 graphs | -- |
| | MovieNet | ECCV 2020 | 2007.10937 | 1100 movies 1.1M chars 42K scenes | Reference |
| | DramaQA | AAAI 2021 | 2005.03356 | 4-level cog hier K-drama | 17983 QA |
| | TVQA | EMNLP 2018 | 1809.01696 | 152K QA 6 TV 460h | Pre-MLLM |
| | NarrativeQA | TACL 2018 | 1712.07040 | Book/script | ROUGE-L ~58% |
| | CHIRON | EMNLP-F 2024 | 2406.10190 | Char-sheet mask predict | -- |
| | MovieChat-1K | CVPR 2024 | 2307.16449 | 1K long vid | -- |
| | CinePile | NeurIPS 2024 | 2405.08813 | 305K MCQ from AD+LLM | -- |
| | MovieCORE | EMNLP 2025 Oral | 2508.19026 | System-2 cog QA | ACE +25% |
| | **SeriesBench** | CVPR 2025 | 2504.21435 | 105 drama 28 tasks | -- |
| | VRBench | ICCV 2025 | 2506.10857 | 1010 vid avg 1.6h 9468 QA + 30292 steps | 7 reasoning types |
| **人格/Persona** | PsychoBench | ICLR 2024 | 2310.01386 | OCEAN 13 scales | LLMs partly OCEAN-stable |
| | EQ-Bench | 2023 | 2312.06281 | 60 Q emo intensity | r=0.97 with MMLU |
| | EmoBench (Sabour) | 2024 | 2402.12071 | EI EN+ZH 400 Q | (above) |
| | PersonaChat | 2018 | 1801.07243 | Persona dialogue | -- |
| | DialogRE | ACL 2020 | 2004.08056 | 36 relations Friends | (above) |
| | CharacterEval | 2024 | 2401.01275 | 13 metrics × 4 dim 中文 | 11376 ex / 77 chars |
| | RoleLLM/Bench | 2023 | 2310.00746 | 100 roles 168K samples | RoleLLaMA ≈ GPT-4 |
| | OpenCharacter | 2025 | 2501.15427 | Customizable role-play 合成 persona | -- |
| | Big5-Chat | 2024 | 2410.16491 | Big5-grounded persona training | -- |

### 3.3 三个最直接竞品的深度对比

#### 3.3.1 CharToM-QA（最直接文本竞品）
- **论文**：Zhou et al. arXiv 2501.01705, ACL 2025 Long
- **核心论点**：ToM 不止本地 belief tracking，需要长 history/personal context（背景、人格、过往互动、社会关系）
- **构造**：挖经典小说**读者笔记**（marginalia）→ 结构化 ToM QA
- **规模**：20 部经典小说 × 1035 问题（每书 15-167 问），4 维：Belief / Intention / Emotion / Desire
- **格式**：生成式（BPC + Penalty Rate）+ 4-选 MCQ
- **结果**：
  - 人类 **familiar 59.3-62.0%** vs **unfamiliar 41.3-48.7%**（≈21pp gap 完全来自长上下文）
  - GPT-4o 51.8-58.7%；DeepSeek-R1 54.0%；o1 51.3%
- **与本项目共同点**：四维心理状态、强调长背景、角色中心、novel-grade rich narratives
- **关键差异（本项目优势）**：
  1. **文本-only** → 我们多模态视频
  2. **无 episodic continuity** → 我们跨集 + 多 horizon
  3. **静态 dim 无关系动态** → 我们 T10 contagion 通过 M6 网络
  4. **OCEAN 隐式 context** → 我们 T9 显式测量
  5. **预训练污染**（经典小说 in pretraining）→ 我们用 post-cutoff TV

#### 3.3.2 HumanVBench（方法学基础）
- **论文**：Zhou et al. arXiv 2412.17574, CVPR 2026
- **任务**：16 任务分两 dim
  - **Inner Emotion (5)**：ER, Emotion Temp Analysis, Attitude Recog, Emotion Intensity Comparison
  - **Outer Manifestations (11)**：Person Recog (text↔human, count, time-interval, 4 子)；Behavior Analysis (temporal, causality, at-specified-time, time-of-action, 4 子)；Speech-Visual Alignment (AV speaker matching, ASD, AV align, speech-content matching, 4 子)
- **规模**：2116 MCQ over 4.7h copyright-free Pexels
- **Filter & 合成 pipeline（我们复用）**：
  1. Resolution ≥1280×480, scene-based seg, duration ≥1s, motion ≥1.2, face visibility ≥0.65
  2. S3FD 人脸 tracking + DeepFace demographic + ASD-Light + SenseVoice ASR + VideoLLaMA-2.1 emotion
  3. **Distractor-included QA pipeline**：(1) 选 video by task → (2) 生成 Q/A with marked vid/face crops → (3) 3 个独立 MLLM 迭代优化答案 → (4) 从被舍弃的候选生成 distractor → (5) 手工 verify
  4. 6% 移除（answer leakage）
- **结果**（22 SOTA MLLM）：Gemini-1.5-Pro 65.5% best；VideoLLaMA-3 58.4%；GPT-4o 51.7%；human 89.1%
- **关键 finding**：surprise misclassification ~41% of top-performer emotion mistakes；temporal reasoning 显式 timestamp 有帮助
- **与本项目差异**：
  - 短 Pexels clip 而非 TV 集 + 跨集
  - 无 ToM / false-belief / intent dim
  - 无 OCEAN
  - Speech-visual 比重过大，higher-order ToM 不足

#### 3.3.3 MoMentS（最直接视频 ToM 对手）
- **论文**：Villa-Cueva et al. arXiv 2507.04415, EMNLP-F 2025
- **任务**：ATOMS 完整 taxonomy 7 维：Beliefs / Desires / Intentions / Emotions / Knowledge / Percepts / Non-Literal Communication
- **数据**：168 短片（avg 14.56 ± 4.65 min；144 English + 11 non-English）from SF20K（Omeleto shorts）+ GPT-4o 用 synopsis 选 socially rich films
- **规模**：2344 MCQ（4 options each）= 9376 candidate；16 标注员（多为心理本科）× 6 周
- **创新**：
  - 每周交替 question writing vs distractor writing
  - LLM-in-the-loop distractor evaluator（实时打分 distractor set bias）→ pilot 后减少 bias 19-24pp
  - 总标注成本 ~$8,745
- **结果**：LLaVA-Video-72B **67.66%**（multimodal）/ 62.1%（text）；InternVL2.5-8B 51.79%；human 86%；vision 加 3-10pp typically；模型 underutilize visual
- **与本项目差异**：
  - **短片 ~14min vs TV 集 40-60min + season arcs 5-20h**
  - **无跨片角色连续性** — short films 都是独立的
  - **OCEAN 视为隐式 context 而非测量构念** → 我们 T9 显式
  - **Emotions 是 per-char** → 我们 T10 contagion 是 graph propagation
  - 7 ATOMS abilities vs 我们 9 task 更广

### 3.4 视频 ToM 整体空白

- **Long-horizon 角色连续性 across episodes**：完全空白 → T9 填补
- **OCEAN 从持续视频**：text-only 文献存在（PsychoBench, Big5-Chat, RoleLLM, CharacterEval）；视频完全空白 → T9 首个
- **情感传染作为 graph propagation 任务**：完全空白 → T10 首个
- **多角色 relational ToM at scale**：MuMA-ToM 2-4 agents 合成；SIV-Bench Relation Inference 仅 bottleneck → T10 提供 graph 结构
- **Applied vs explicit ToM on long video**：SimpleToM gap 在文本，视频未测试 → T9 + T6/T7 联动测试
- **Causal counterfactual over narrative arcs**：CRASS/Eyes-Can-Deceive 静态 text/image → T6 multi-hop 视频版
- **Higher-order ToM in video**：HiToM 仅文本 → 可延伸
- **预训练未泄露内容**：经典小说 contaminated（CharToM-QA 承认），post-2023 TV 缓解
- **传染方向性**：MELD/IEMOCAP/MERR per-char label → 我们强制 directionality
- **统一 taxonomy**：(a) ATOMS-style ToM (MoMentS-aligned)、(b) personality (PsychoBench-style for video)、(c) relational influence (novel)、(d) narrative continuity (SeriesBench-aligned)、(e) emotional reasoning (HumanVBench-aligned) — 无 prior 提供统一

---

## 4. Video Agent 与长视频记忆系统

### 4.1 三大范式

1. **End-to-end Long-context Video MLLM**：LongVA、LongVU、LongVILA、Video-XL、VideoChat-Flash、InternVideo2.5、Qwen2.5-VL、Qwen2.5-Omni。1M+ token 或 2K-10K 帧通过 token 压缩、层次注意、序列并行、KV 稀疏化。
2. **Memory-Augmented MLLM**：MA-LMM、MovieChat/+、VideoLLaMB、MC-ViT、∞-Video、AdaCM²、HEM-LLM。把显式 memory 模块集成进 forward pass。
3. **Tool-Using Video Agent**：**M3-Agent**、VideoAgent (Wang/Fan)、DoraemonGPT、TraveLER、OmAgent、VideoTree、DrVideo、Vgent、VideoExplorer、VideoDeepResearch、Deep Video Discovery、WorldMM、GCAgent、VideoForest、LongVideoAgent、VideoARM、Video-EM。LLM 控制器 + 结构化记忆 + 工具调用。

### 4.2 记忆架构趋势

- **Flat token 库** → **hierarchical / event-based** → **entity-centric graphs**
- **纯 episodic** → **episodic + semantic**（Generative Agents 风格）
- **single-shot CLIP retrieval** → **iterative agentic search 多工具编排**

### 4.3 关键系统简表

| 类 | 系统 | 年份 | arXiv | 记忆架构 | Backbone | Best |
|---|---|---|---|---|---|---|
| **Video Agent** | **M3-Agent** | 2025 | 2508.09736 | **Entity-centric multimodal graph**: episodic + semantic; face_id/voice_id/character_id linkage | Qwen2.5-Omni-7B + Qwen3-32B (DAPO RL) | M3-Bench-robot/web, V-MME-long; +6.7/7.7/5.3% over Gemini-1.5-Pro/GPT-4o |
| | WorldMM | 2025 | 2512.02425 | Episodic + Semantic + Visual; adaptive retrieval | -- | 5 LVQA; +8.4% over SOTA |
| | **GCAgent** | 2025 | 2511.12027 | **Schematic + Narrative episodic** w/ causal+temporal edges | Qwen2.5-VL 7B | V-MME Long 73.4% (+23.5%) |
| | VideoAgent (Wang) | ECCV 2024 | 2403.10517 | CLIP frame caption + retrieval | GPT-4 + EVA-CLIP + VLM | EgoSchema 54.1%, NExT-QA 71.3% zero-shot |
| | VideoAgent (Fan) | ECCV 2024 | 2403.11481 | Temporal event + object-centric tracking | LLM agent | EgoSchema +26%, NExT-QA +6.6% |
| | DoraemonGPT | ICML 2024 | 2401.08392 | Space-dominant (instances) + time-dominant; MCTS | LLM + tools | Various |
| | TraveLER | EMNLP 2024 | 2404.01476 | Frame-level info store; Traverse-Locate-Evaluate-Replan | LMM agents | SOTA zero-shot 4 benchmarks |
| | OmAgent | EMNLP 2024 | 2406.16620 | Video2RAG preprocessor + divide-and-conquer | LLM + MMs | 24h videos; OmAgent-Bench 2000+ |
| | VideoTree | CVPR 2024 | 2405.19209 | Hierarchical tree adaptive breadth+depth | LLM + CLIP | EgoSchema +7% beats GPT-4V on V-MME-long |
| | DrVideo | CVPR 2025 | 2406.12846 | Video→coarse doc + iterative agent | LLM + captioning | MovieChat global +38, +30.2 LLama-Vid-QA |
| | Vgent | 2025 | 2510.14032 | Video graph w/ semantic | LVLM | MLVU +3-5.4% |
| | Deep Video Discovery | NeurIPS 2025 | 2505.18079 | Multi-granular DB + search tools | LLM agent | **LVBench 74.2%/76.0% w/ transcripts** |
| | VideoDeepResearch | 2025 | 2506.10821 | Text-only LRM + multimodal toolkit | LRM + tools | MLVU +9.6, LVBench +6.6 |
| | VideoExplorer | 2025 | 2506.10821 | Sub-question + moment + task percep | LVLM | MLVU 55.4% |
| | **LongVideoAgent** | 2025 | 2512.20618 | Multi-agent: master+grounding+vision; episode-level | Multi-agent | LongTVQA / LongTVQA+ |
| | VideoForest | ACM MM 2025 | 2508.03039 | Person-anchored hier via ReID/tracking | Multi-agent | CrossVideoQA 71.93% person, 83.75% behav |
| | Video-EM | 2025 | 2508.09486 | Training-free key event + episodic + CoT | Video-LLM-agnostic | LVU |
| **Mem-Aug** | MA-LMM | CVPR 2024 | 2404.05726 | Visual + Query mem banks online | InstructBLIP-style | LVU 63.0% |
| | MovieChat / + | CVPR 2024 | 2307.16449 | Atkinson-Shiffrin short+long | LLM + ViT | >10K frame; MovieChat-1K |
| | VideoLLaMB | ICCV 2025 | 2409.01071 | Recurrent memory bridge + SceneTiling | LLaMA-3-8B | +5.5 over competitors |
| | MC-ViT / ∞-Video | ICLR 2024/25 | 2402.05861/2501.19098 | Non-parametric consolidation ViT activations | ViT | EgoSchema/PT/Diving48 SOTA |
| | HEM-LLM | 2024 | 2409.06299 | Event-based local + global | Video-LLM | LVQA |
| | AdaCM² | CVPR 2025 | 2411.12593 | Adaptive cross-modality reduction | Video-LLM | >2h |
| | LangRepo | ACL 2025 | 2403.14622 | All-textual repo, prune-on-redundancy | LLM zero-shot | EgoSchema/NExT-QA SOTA |
| | LifelongMemory | 2024 | 2312.05269 | Concise textual log + Caption Digest | LLM | EgoSchema SOTA |
| | Goldfish | ECCV 2024 | 2407.12679 | Per-clip detailed desc + top-k retrieval | MiniGPT4-Video | **TVQA-long 41.78% (+14.94%)** |
| **Long-ctx MLLM** | LongVA | 2024 | 2406.16852 | None; pure long-ctx 2K frame | Qwen2 | V-MME SOTA at 7B |
| | LongVU | 2024 | 2410.17434 | DINOv2 compr → 8K | LLaMA/Qwen | V-MME 60.9 7B |
| | LongVILA | 2024 | 2408.10188 | NVIDIA MM-SP seq parallel | VILA + LLaMA | NIAH 99.8% @6000 frames |
| | Video-XL | 2024 | 2409.14485 | VST + KV sparse, 2048 frame | LLM | VNBench +10% |
| | Video-XL-Pro / 2 | 2025 | 2503.18478/2506.19225 | Reconstructive tok compr / KV-aware | LLM | Longer & cheaper |
| | VideoChat-Flash | ICLR 2026 | 2501.00574 | HiCo 1/50 ratio, 10K frame NIAH 99.1% | InternVL backbone | MLVU 74.5% SOTA 7B |
| | InternVideo2.5 | 2025 | 2501.12386 | DPO + adaptive hier compr | InternVL | 6× longer ctx |
| | Qwen2.5-VL | 2025 | 2502.13923 | Win-attn + dynamic FPS + abs-time MRoPE | Qwen2.5 | Multi-hour, sec-level localization |
| | Qwen2.5-Omni | 2025 | 2503.20215 | Thinker-Talker streaming + TMRoPE | Qwen2.5 | OmniBench SOTA |
| | TimeChat / -Online | CVPR 2024 / ACM MM 2025 | 2312.02051/2504.17343 | Time-aware Q-Former / DTD streaming | LLM + ViT | Charades-STA +27.5; StreamingBench 98% w/ -82.8% tok |
| **Text Agent ref** | Generative Agents | UIST 2023 | 2304.03442 | Memory stream + reflection + plan; recency × importance × relevance | GPT-3.5/4 | 25-agent Smallville |
| | MemGPT | ICLR 2024 | 2310.08560 | OS-inspired tiered + function-call paging | GPT-4 | Letta |
| | MemoryBank | AAAI 2024 | 2305.10250 | Daily chat + summary + **user personality/mood**; Ebbinghaus decay | LLM | SiliconFriend |
| | A-MEM | 2025 | 2502.12110 | Zettelkasten notes + agentic link + evolution | 6 LLMs | LongMemEval SOTA |
| | Reflexion | NeurIPS 2023 | 2303.11366 | Verbal self-reflection buffer | LLM | HotpotQA/AlfWorld/HumanEval |
| | Voyager | 2023 | 2305.16291 | Ever-growing skill library | GPT-4 | Minecraft lifelong |
| **Event graph** | MECD+ | NeurIPS 2024/25 | 2501.07227 | Event-level causal graph + Granger | LLM + video | Causal-VidQA |
| | GraphThinker | 2026 | 2602.17555 | Event-based scene graph (EVSG) intra/inter | MLLM | Video reasoning |

### 4.4 M1-M7 vs 现有系统对齐表

| 系统 | M1 perception | M2 episode DAG | M3 cross-ep DAG | M4 emo traj | M5 OCEAN | M6 relation | M7 world |
|---|---|---|---|---|---|---|---|
| M3-Agent | ✓ | 部分 | ✗ | ✗ | 部分 (entity) | 部分 (id-link) | 部分 (semantic) |
| WorldMM | ✓ | 部分 | ✗ | ✗ | ✗ | ✗ | ✓ semantic |
| GCAgent | ✓ | **✓** causal+temporal | ✗ | ✗ | 部分 (role) | ✗ | 部分 (situation) |
| VideoAgent (Fan) | ✓ | 部分 | ✗ | ✗ | ✗ | 部分 (object) | ✗ |
| VideoForest | ✓ | ✗ | 部分 (person) | ✗ | ✗ | 部分 (traj) | ✗ |
| LongVideoAgent | ✓ | ✗ | 部分 (episode) | ✗ | ✗ | ✗ | ✗ |
| MECD+ / GraphThinker | ✓ | **✓** causal | ✗ | ✗ | ✗ | ✗ | ✗ |
| MovieChat / MA-LMM | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Generative Agents (text) | N/A | N/A | N/A | 部分 (mood) | **✓** | 部分 (agent-agent) | 部分 (town) |

**关键观察**：**没有任何系统同时覆盖 M3+M4+M5+M6+M7**。M3-Agent 和 GCAgent 在结构上最接近（entity graph + event graph），但都不抽人格 trait 或情感轨迹。Generative Agents 拥有正确的 persona/memory 范式但是文本端 2 天模拟。

### 4.5 M3-Agent 深度对比（关键 baseline）

- **作者**：ByteDance-Seed, arXiv 2508.09736, 2025
- **记忆架构**：entity-centric multimodal graph
  - 节点：(id, modality, raw content, embedding, weight, metadata)
  - 边：undirected; 身份通过 face_id / voice_id / character_id linkage 锚定（避免文本歧义）
  - 两类：**episodic**（concrete events）+ **semantic**（consolidated world knowledge）
- **Backbone**：Qwen2.5-Omni-7B for memorization（imitation learning, 10952 samples）+ Qwen3-32B for control（DAPO RL, GPT-4o reward）
- **推理**：up to H=5 rounds of [Search] (search_node / search_clip) and [Answer]
- **评估**：M3-Bench-robot（100 robot 视角）+ M3-Bench-web（920 web 视频）+ Video-MME-long
- **性能**：相比 Gemini-1.5-Pro + GPT-4o 的 prompting agent **+6.7/+7.7/+5.3pp**
- **开源**：github.com/ByteDance-Seed/m3-agent

**M3-Agent 与我们 M1-M7 的对应**：M1 ✓（real-time perception via Qwen2.5-Omni），M2 部分（memory graph 但无显式 DAG 因果边），M3 ✗（单视频），M4 ✗（无 emotion traj），M5 部分（entity-centric 但非 OCEAN），M6 部分（entity link 但非 typed relation），M7 部分（semantic memory 但非 propositional 世界状态）。

### 4.6 缺口

1. **跨集 evaluation**：仅 LongTVQA/+ 显式多集；其他系统全部单视频
2. **Emotion trajectory**：MME-Emotion 单 clip；MTMEUR 多轮但单话题
3. **OCEAN persona extraction from video**：text-only 文献丰富，视频空白
4. **Relationship network as graph**：VideoForest 人物 ID 而非 typed 关系
5. **World/plot state**：GCAgent/WorldMM 最接近但非 propositional 事实
6. **Modular memory-injection (E3 setting)**：无 benchmark 暴露 pre-extracted memory 为受控输入
7. **End-to-end MLLM vs Agent 公平对比**：当前 apples-to-oranges
8. **Multi-hop temporal-causal reasoning**：CausalStep, SeriesBench 触及但 shorter content

---

## 5. 视频 MLLM 模型生态（2024-2026）

### 5.1 三个迭代波

- **2024 wave**：Video-LLaVA、Video-ChatGPT、MiniGPT4-Video、VILA-1.5、LLaVA-OneVision、LLaVA-Video、Video-LLaMA2/3。CLIP encoder + projector + LLM；16-64 帧；V-MME 55-65%。
- **2025 wave**：Qwen2.5-VL、InternVL3、InternVideo2.5、VideoChat-Flash、LongVU、LongVILA、Apollo、Video-XL/Pro/2、NVILA、VideoLLaMA3。1024-10000 帧通过 hierarchical token 压缩；开源破 70% V-MME。
- **2026 wave**：Qwen3-VL、Qwen3-Omni、Qwen3.5-Omni、Gemini 3 Pro / 3.1 Pro、GPT-5/5.2/5.5、Claude Opus 4.7、MiniCPM-V/o 4.5、video-SALMONN 2、Kimi K2.5/K2.6。原生 1M-2M token context（2+ 小时视频）+ unified audio-visual reasoning。

### 5.2 截至 2026-05 的 Video-MME 排行

| 模型 | V-MME score | 备注 |
|---|---|---|
| Kimi K2.5 (proprietary MoE) | **87.4%** | leaderboard 领跑 |
| Gemini 2.5 Pro | **84.8%** (w/sub) | 1M token + native audio |
| Qwen3.6 Plus (proprietary) | 84.2% | -- |
| Gemini 3 Pro | 77.3% Instruct | Video-MMMU 87.6% |
| GPT-5.5 long-form | 71.2% | Realtime audio |
| Qwen 3.5-Omni | 69.5% | -- |

### 5.3 音频处理三类（关键于情感）

1. **Native waveform**：Gemini 2.5/3 native audio、GPT-4o/5 Realtime、Qwen2.5/3/3.5-Omni、MiniCPM-o 4.5、video-SALMONN/SALMONN-2/SALMONN-S、NExT-GPT、InteractiveOmni
2. **ASR-then-text**：Qwen2.5-VL、Qwen3-VL、InternVL3、LLaVA-Video、VideoChat-Flash
3. **Vision-only**：Apollo、Video-XL、LongVU、MiniGPT4-Video、Pixtral、Claude Opus 4.x

**对情感推理至关重要**：只有 (1) 保留 prosody（音调、颤抖、停顿、叹息）。MTV "I'm fine"（平淡 vs 颤抖）在转录后完全相同 — 仅看字幕的模型必然漏过 paralinguistic 线索。

开源 paralanguage leader：**Qwen3-Omni**（20M-h audio encoder from scratch）+ **video-SALMONN 2-72B**（在 AV-aware Video-MME 上超 GPT-4o）。闭源 leader：**Gemini 2.5/3 Pro**（native audio）。

### 5.4 我们的推荐评测 slate

**Tier 1 闭源（must-have）**：
1. **Gemini 2.5 Pro** — 主闭源 baseline；6h 视频 + native audio；~$200-400 for 1800 QA
2. **GPT-5 / 5.5** — 强 reasoning + Realtime audio；128-400K context；~$600-1200

**Tier 2 开源 AV（must-have for emotion）**：
3. **Qwen3-Omni-30B-A3B** — Apache 2.0；native waveform；22/36 AV bench SOTA；1×A100 5-10 GPU-h/setting；~$300-600
4. **video-SALMONN 2-72B** — 学术开源 AV 顶配；speech + audio + visual；2×H100；~$800-1500

**Tier 3 开源 VL**：
5. **Qwen3-VL-235B-A22B-Thinking** — 顶级开源 VL；2h native；4-8×H100；~$1500-3000
6. **InternVL3.5-78B** — V2PE 长 context；2-4×H100；~$1000-1800

**Tier 4 长视频专家**：
7. **VideoChat-Flash-7B** — 16 tok/frame；3h；最便宜；1×A100；~$100-200

**Optional**：LongVU-7B（hour-scale）、Apollo-7B（Meta）、MiniCPM-o 4.5（on-device omni）、Goldfish+MiniGPT4-Video（TV-show targeted retrieval baseline）

**总预算估算**：闭源 API $800-1600 + 开源 GPU $3700-7100 = **$4500-8700** 完整 7-模型 × 1800 QA × 4 settings。最小可行 slate（Gemini 2.5 Pro + GPT-5 + Qwen3-Omni + Qwen3-VL + video-SALMONN-2-7B）：**$1500-2500**。

### 5.5 Audio-for-emotion 深度

情感视频中信号 = 视觉（表情、姿态） + 词汇（字幕） + **paralinguistic acoustic features**（韵律 prosody = 音高 contour、intonation；节奏 = 停顿、语速；声音质量 = breathy、tremor；非语言 vocalization = 叹息、笑声、抽泣）。仅看字幕系统损失 paralinguistic（10-20pp 损失 on IEMOCAP，Omni-Emotion arXiv 2501.09502）。

**实战含义**：本 benchmark 应包含至少一类显式要求 paralinguistic 推理的任务子集（lexical 中性但 prosody 不中性）。对仅有视觉+字幕的模型，提供两条评测路线：(i) 字幕 only；(ii) 字幕 + audio-tagged metadata（"[sobbing]" 标签）— 量化 audio gap。

---

## 6. 综合定位与本项目论文的卖点矩阵

### 6.1 与领域整体的差异化（更新版）

| # | 卖点 | 现状最接近工作 | 本项目独占性 |
|---|---|---|---|
| 1 | **首个 9-任务 × 4-Setting 矩阵的情感长视频推理 benchmark** | MTMEUR（3 任务 × 1 setting）、MME-Emotion（8 任务 × 1 setting）、HumanVBench（16 任务 × 1 setting） | 9 × 4 = 36-cell 评估矩阵；显式分层 memory regime |
| 2 | **首次显式区分 M1-M7 7 个记忆组件 + 任务-记忆依赖矩阵** | M3-Agent（entity graph + episodic + semantic 部分）、GCAgent（schematic + narrative） | M4 + M5 + M6 + M7 完整覆盖；OCEAN trait inference 显式 |
| 3 | **新任务 T7/T8 情感→行动预测**：video-only 才能做 | MTMEUR future-action 子任务；CausalVQA anticipation/planning physical | 跨集长 horizon（30min/cross-ep）首次 |
| 4 | **新任务 T9 个性一致性**：依赖 M5 语义记忆 | PsychoBench/Big5-Chat/CharacterEval 全 text-only；First Impressions V2 15s clip；CPED utt 级 | 首个 video × OCEAN × 长 horizon × consistency |
| 5 | **新任务 T10 情感传染**：M6 关系网络 + 跨角色情感影响 | MELD/IEMOCAP per-char label；社交媒体情感传染建模 | 首个 AI benchmark，强制 directionality |
| 6 | **全自动 pipeline + F7 M-依赖过滤器** | HumanVBench distractor + filter pipeline；MoMentS LLM-in-the-loop distractor evaluator | F7 M-依赖检查首创 |
| 7 | **协同评估**：与三篇论文的另两篇 Video Agent 在 M5/M6/M7 schema 上原生对齐 | M3-Agent + WorldMM + GCAgent | 模块化 memory-injection E3 setting 首创 |
| 8 | **中英双语并行 + 季级 scale** | Movie101（仅中文电影 pre-MLLM）、TVQA（仅英文）、M3ED（中文 utt） | 中英 6 部 × 10 集首创 |
| 9 | **集成 quality 控制 7 道 filter** | HumanVBench 5-step + 6% leak；MoMentS LLM evaluator | F1-F7 涵盖 no-context、subtitle-only、modal-ablation、cross-model consensus、option-swap、future-leak、M-dependency |
| 10 | **跨集情感连续 + multi-hop causal chain** | RECCON/ECPE single-hop；MTMEUR 单话题多轮；VRBench 多步但 1.6h 非情感 | T4 转折定位 + T5 单跳 + T6 多跳一体 |

### 6.2 与最近工作的具体对比要点（可以直接进 paper Related Work）

| 与谁比 | 共享 | 我们的独占 |
|---|---|---|
| **CharToM-QA** | 4 心理维度、长背景重要、角色中心 | 多模态视频、跨集 continuity、OCEAN 显式、relational dynamics、post-cutoff 数据 |
| **HumanVBench** | filter pipeline、distractor 合成、emotion + face track | TV-集级 vs Pexels 短片、ToM/intent/persona、relational/contagion |
| **MoMentS** | ATOMS taxonomy、MCQ、LLM-in-loop distractor | 季级 TV vs 14min short film、character continuity 跨片、OCEAN trait inference、emotion contagion graph、9 任务覆盖更广 |
| **MTMEUR** | 多轮 emo+cause+future action 主题 | clip <1min vs 季级、多 horizon 行动预测、OCEAN/contagion 独占 |
| **MME-Emotion** | 8 emotion tasks + CoT scoring | clip 级 vs 季级、+T6/T7/T8/T9/T10 新任务、4-setting matrix |
| **SeriesBench** | 105 drama series TV narrative | 28 narrative tasks vs 我们 9 emotional/ToM tasks、显式 memory regime、E3 setting、跨集 mem injection |
| **VRBench** | 多步推理 + 时间戳 reasoning steps | 1.6h vs 季级、generic narrative 而非情感、无 OCEAN/contagion |
| **M3-Agent** | Entity-centric memory + episodic + semantic + face_id/voice_id/char_id | M3/M4/M5/M6/M7 完整 vs partial、cross-episode vs single-video、benchmark vs system |

### 6.3 推荐 baseline 集（更新版）

#### Video Agent baselines
- **M3-Agent**（KEY，arxiv 2508.09736，code github.com/ByteDance-Seed/m3-agent）
- GCAgent（2511.12027）
- WorldMM（2512.02425）
- Deep Video Discovery（2505.18079，code github.com/microsoft/DeepVideoDiscovery）
- VideoDeepResearch（2506.10821）
- VideoExplorer（2506.10821）
- VideoAgent (Wang) / (Fan)（2403.10517/2403.11481）
- DoraemonGPT（2401.08392）
- TraveLER（2404.01476）
- OmAgent（2406.16620）
- VideoTree（2405.19209）
- DrVideo（2406.12846）
- Vgent（2510.14032）
- LongVideoAgent（2512.20618，跨集 reference）
- VideoForest（2508.03039，cross-video reference）
- 文本端 reference：Generative Agents（在抽取的字幕/事件 caption 上跑）、MemGPT、A-MEM

#### End-to-end MLLM baselines（按重要性）
- Tier 1：Gemini 2.5 Pro、GPT-5/5.5
- Tier 2：Qwen3-Omni-30B-A3B、video-SALMONN 2-72B
- Tier 3：Qwen3-VL-235B、InternVL3.5-78B、Qwen2.5-Omni-7B、Qwen2.5-VL-72B
- Tier 4：VideoChat-Flash-7B、Apollo-7B、LongVU-7B、Goldfish+MiniGPT4-Video（TV-targeted retrieval baseline）
- Optional：MiniCPM-o 4.5、InternVideo2.5、Video-XL-2

#### Emotion-specialized baselines（为情感任务定位）
- Emotion-LLaMA（2406.11161）
- Emotion-LLaMAv2 + MMEVerse（2601.16449）
- AffectGPT（2501.16566）
- R1-Omni（2503.05379）
- AVERE / EmoReAlM（2602.07054）
- VidEmo（2511.02712）

#### Text-only baselines（消融用）
- GPT-5 字幕-only
- Qwen3-32B 字幕-only
- 验证字幕泄露 / 视觉必要性

---

## 7. 关键参考文献（按主题）

### 7.1 长视频 benchmark
- Video-MME — Fu et al. CVPR 2025, **arXiv:2405.21075**
- MLVU — Zhou et al. CVPR 2025, **arXiv:2406.04264**
- LongVideoBench — Wu et al. NeurIPS 2024 D&B, **arXiv:2407.15754**
- MVBench — Li et al. CVPR 2024, **arXiv:2311.17005**
- EgoSchema — Mangalam et al. NeurIPS 2023 D&B, **arXiv:2308.09126**
- HourVideo — Chandrasegaran et al. NeurIPS 2024 D&B, **arXiv:2411.04998**
- LVBench — ICCV 2025, **arXiv:2406.08035**
- InfiniBench — EMNLP 2025, **arXiv:2406.19875**
- CG-Bench — ICLR 2025, **arXiv:2412.12075**
- TVBench — BMVC 2025, **arXiv:2410.07752**
- TempCompass — ACL-F 2024, **arXiv:2403.00476**
- CinePile — NeurIPS 2024 D&B, **arXiv:2405.08813**
- MovieChat-1K — CVPR 2024, **arXiv:2307.16449**
- MovieNet — ECCV 2020, **arXiv:2007.10937**
- MoVQA — 2023, **arXiv:2312.04817**
- MovieCORE — EMNLP 2025 Oral, **arXiv:2508.19026**
- MF² (Movie Facts and Fibs) — 2025, **arXiv:2506.06275**
- **SeriesBench** — CVPR 2025, **arXiv:2504.21435**
- **VRBench** — ICCV 2025, **arXiv:2506.10857**
- LongTVQA / LongVideoAgent — 2025, **arXiv:2512.20618**
- TVQA — EMNLP 2018, **arXiv:1809.01696**; TVQA+ — ACL 2020, **arXiv:1904.11574**
- NExT-QA — CVPR 2021, **arXiv:2105.08276**; NExT-GQA — CVPR 2024 Highlight, **arXiv:2309.01327**
- CausalVQA (Meta) — 2025, **arXiv:2506.09943**
- Perception Test — NeurIPS 2023 D&B, **arXiv:2305.13786**
- VideoNIAH — 2024, **arXiv:2406.09367**
- VideoEval-Pro — 2025, **arXiv:2505.14640**
- X-LeBench — EMNLP 2025-F, **arXiv:2501.06835**
- Movie101 — ACL 2023, **arXiv:2305.12140**
- Ego4D — CVPR 2022, **arXiv:2110.07058**

### 7.2 多模态情感分析
- MELD — Poria et al. ACL 2019, **arXiv:1810.02508**
- M3ED — ACL 2022, **arXiv:2205.10237**
- **CPED** (含 OCEAN) — 2022, **arXiv:2205.14727**
- DFEW — ACM MM 2020, **arXiv:2008.05924**
- FERV39k — CVPR 2022, **arXiv:2203.09463**
- MAFW — ACM MM 2022, **arXiv:2208.00847**
- Aff-Wild2 / ABAW — **arXiv:2106.15318**
- CAER — ICCV 2019, **arXiv:1908.05913**
- VEATIC — WACV 2024, **arXiv:2309.06745**
- EmoSet — ICCV 2023, **arXiv:2307.07961**
- ECPE — ACL 2019 Outstanding, **arXiv:1906.01267**
- SemEval-2024 T3 MECPE — **arXiv:2405.13049**
- EmoTrigger — **arXiv:2311.09602**
- EIBench (Why-We-Feel) — CVPR-W 2025, **arXiv:2504.07521**
- EmoBench — Sabour et al. ACL 2024, **arXiv:2402.12071**
- **EmoBench-M** — 2025, **arXiv:2502.04424**
- **MME-Emotion** — 2025, **arXiv:2508.09210**
- **MTMEUR** — ACM MM 2025, **arXiv:2508.16859**
- EMER — 2023, **arXiv:2306.15401**
- OV-MER — 2024, **arXiv:2410.01495**
- AffectGPT — ICML 2025 Oral, **arXiv:2501.16566**
- **Emotion-LLaMA** — NeurIPS 2024, **arXiv:2406.11161**
- Emotion-LLaMAv2 + MMEVerse — **arXiv:2601.16449**
- R1-Omni — **arXiv:2503.05379**
- AVERE / EmoReAlM — ICLR 2026, **arXiv:2602.07054**
- CA-MER — ACM MM 2025, **arXiv:2508.01181**
- MERBench — **arXiv:2401.03429**
- VidEmo — **arXiv:2511.02712**

### 7.3 ToM / 叙事 / 角色
- ToMi — EMNLP 2019
- SocialIQA — **arXiv:1904.09728**
- FANToM — **arXiv:2310.15421**
- HiToM — **arXiv:2310.16755**
- OpenToM — **arXiv:2402.06044**
- ToMBench — **arXiv:2402.15052**
- SimpleToM — **arXiv:2410.13648**
- ExploreToM — **arXiv:2412.12175**
- DynToM — **arXiv:2505.17663**
- **CharToM-QA** — ACL 2025, **arXiv:2501.01705**
- Social-IQ / Social-IQ 2.0 — CVPR 2019 / ICCV-W 2023
- MMToM-QA — ACL 2024 Outstanding, **arXiv:2401.08743**
- MuMA-ToM — AAAI 2025 Oral, **arXiv:2408.12574**
- **HumanVBench** — CVPR 2026, **arXiv:2412.17574**
- **MoMentS** — EMNLP-F 2025, **arXiv:2507.04415**
- EgoToM — **arXiv:2503.22152**
- SIV-Bench — **arXiv:2506.05425**
- Social Genome — EMNLP 2025, **arXiv:2502.15109**
- Mind the Motions — **arXiv:2511.15887**
- MovieGraphs — CVPR 2018, **arXiv:1712.06761**
- DramaQA — AAAI 2021, **arXiv:2005.03356**
- NarrativeQA — TACL 2018, **arXiv:1712.07040**
- DialogRE — ACL 2020, **arXiv:2004.08056**
- CharacterEval — 2024, **arXiv:2401.01275**
- RoleLLM / RoleBench — **arXiv:2310.00746**
- OpenCharacter — **arXiv:2501.15427**
- Big5-Chat — **arXiv:2410.16491**
- PsychoBench — ICLR 2024, **arXiv:2310.01386**
- EQ-Bench — **arXiv:2312.06281**
- EmoTx — CVPR 2023, **arXiv:2304.05634**

### 7.4 Video Agent / 记忆系统
- **M3-Agent** — 2025, **arXiv:2508.09736**, github.com/ByteDance-Seed/m3-agent
- WorldMM — **arXiv:2512.02425**
- **GCAgent** — **arXiv:2511.12027**
- VideoAgent (Wang) — ECCV 2024, **arXiv:2403.10517**
- VideoAgent (Fan) — ECCV 2024, **arXiv:2403.11481**
- DoraemonGPT — ICML 2024, **arXiv:2401.08392**
- TraveLER — EMNLP 2024, **arXiv:2404.01476**
- OmAgent — EMNLP 2024, **arXiv:2406.16620**
- VideoTree — CVPR 2024, **arXiv:2405.19209**
- DrVideo — CVPR 2025, **arXiv:2406.12846**
- VLog — CVPR 2025, **arXiv:2503.09402**
- Deep Video Discovery — NeurIPS 2025, **arXiv:2505.18079**
- VideoDeepResearch / VideoExplorer — **arXiv:2506.10821**
- LongVideoAgent — **arXiv:2512.20618**
- VideoForest — ACM MM 2025, **arXiv:2508.03039**
- Vgent — **arXiv:2510.14032**
- MA-LMM — CVPR 2024, **arXiv:2404.05726**
- MovieChat / + — CVPR 2024, **arXiv:2307.16449/2404.17176**
- VideoLLaMB — ICCV 2025, **arXiv:2409.01071**
- MC-ViT / ∞-Video — ICLR 2024 / 2025, **arXiv:2402.05861/2501.19098**
- HEM-LLM — **arXiv:2409.06299**
- AdaCM² — CVPR 2025, **arXiv:2411.12593**
- LangRepo — ACL 2025, **arXiv:2403.14622**
- LifelongMemory — **arXiv:2312.05269**
- Goldfish — ECCV 2024, **arXiv:2407.12679**
- Video-RAG — **arXiv:2411.13093**
- Generative Agents — UIST 2023, **arXiv:2304.03442**
- MemGPT — ICLR 2024, **arXiv:2310.08560**
- MemoryBank — AAAI 2024, **arXiv:2305.10250**
- A-MEM — **arXiv:2502.12110**
- Reflexion — NeurIPS 2023, **arXiv:2303.11366**
- Voyager — **arXiv:2305.16291**
- MECD+ — **arXiv:2501.07227**
- GraphThinker — **arXiv:2602.17555**

### 7.5 视频 MLLM
- Qwen2.5-VL — **arXiv:2502.13923**
- Qwen3-VL — **arXiv:2511.21631**
- Qwen2.5-Omni — **arXiv:2503.20215**
- Qwen3-Omni — **arXiv:2509.17765**
- InternVL3 — **arXiv:2504.10479**
- InternVL3.5 — **arXiv:2508.18265**
- InternVideo2.5 — **arXiv:2501.12386**
- VideoChat-Flash — ICLR 2026, **arXiv:2501.00574**
- LLaVA-Video — **arXiv:2410.02713**
- LLaVA-OneVision — **arXiv:2408.03326**
- VideoLLaMA3 — **arXiv:2501.13106**
- VILA-1.5 / NVILA — **arXiv:2407.14093/2412.04468**
- MiniCPM-V 4.5 / MiniCPM-o 4.5 — **arXiv:2509.18154** + HF model card
- LongVA — **arXiv:2406.16852**
- LongVU — **arXiv:2410.17434**
- LongVILA — **arXiv:2408.10188**
- LongLLaVA — **arXiv:2409.02889**
- Video-XL / Pro / 2 — **arXiv:2409.14485/2503.18478/2506.19225**
- Apollo (Meta) — CVPR 2025, **arXiv:2412.10360**
- TimeChat / -Online — CVPR 2024 / ACM MM 2025, **arXiv:2312.02051/2504.17343**
- video-SALMONN / 2 / S — **arXiv:2406.15704/2506.15220/2510.11129**
- NExT-GPT — ICML 2024, **arXiv:2309.05519**
- Omni-Emotion — **arXiv:2501.09502**

---

## 8. Survey 总结

本 survey 系统调研了 200+ 篇相关工作，覆盖（1）长视频理解 benchmark、（2）多模态情感分析、（3）Theory of Mind 与角色叙事、（4）Video Agent 与记忆系统、（5）视频 MLLM。

**关键洞察**：
- 长视频 benchmark 蓬勃发展但 ≤2h；情感 benchmark 几乎全部 clip 级；ToM benchmark 文本远多于视频；OCEAN 视频几乎空白；情感传染完全空白。
- 最直接的三个竞品 CharToM-QA / HumanVBench / MoMentS 在不同维度逼近我们的工作但都不重叠：CharToM-QA 文本+小说、HumanVBench 短 Pexels clip+方法学、MoMentS 短片+ATOMS 但无 OCEAN/contagion。
- 本项目占据**季级 × 跨集情感 × OCEAN × 传染图**四维交叉的完整空白。
- Video Agent 端 M3-Agent 是最接近的对照系统，但缺 M4/M5/M6/M7 完整；GCAgent 在 event graph 上最优；Generative Agents 是 persona memory 的文本黄金标准但未视频化。
- 视频 MLLM 端，2026-05 闭源 Gemini 2.5/3 Pro + GPT-5/5.5 占据 ceiling；开源 Qwen3-VL/Omni、InternVL3.5、VideoChat-Flash、video-SALMONN-2 是评测重心。

**直接产出**：在第 6 节给出了完整的 baseline slate（Video Agent 15 个 + MLLM 18 个 + emotion-specialized 6 个 + text-only 消融）和成本估算（minimum slate $1500-2500，完整 sweep $4500-8700）。

**下一步**：基于此 survey，第二步将 `emotion_video_bench.md` 计划文档补充以下要素：（a）显式 Related Work 比较表；（b）方法学借鉴明细（HumanVBench filter pipeline、MoMentS distractor evaluator、TVBench shortcut diagnostics）；（c）扩充的 baseline 列表；（d）成本预算；（e）潜在 paralinguistic / audio-tagged evaluation 子集；（f）pretraining contamination 缓解策略。
