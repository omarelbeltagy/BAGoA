# Experiment Log

This file records all experiments run as part of the thesis:
**"Bias-Aware Graph-of-Agents: Measuring, Propagating, and Attributing Social Bias in Multi-Agent LLM Systems"**

---

## 2026-06-02

### Experiment 1 — GoA Baseline Verification (MMLU_sampled)

**Goal:** Verify that the GoA implementation reproduces the paper's reported results before proceeding to bias analysis.

**Infrastructure:** RunPod 6× NVIDIA L40S (48GB VRAM each), one model per GPU.

**Setup notes:**
- RunPod's nginx proxy permanently reserves port 8001 → `qwen_coder` remapped to port 8006 (branch `l40s`).
- Models served in FP16 via vLLM, downloaded to `/workspace/hf_cache` on first run (~20–40 min), cached on subsequent runs (~3 min).

**Command:**
```bash
python main.py \
    --data MMLU_sampled --eval dev \
    --reference_models qwen,qwen_coder,mathstral,biomedical_llama,finance_llama,saul \
    --meta_llm qwen --graph_pooling_method mean \
    --top_k 3 --threshold 0.05 --rounds 1 \
    --temperature 0.7 --max_tokens 800 --num_proc 1 --seed 0
```

**Result:**
| Run | Questions | Accuracy | Paper (GoA_Mean) |
|---|---|---|---|
| MMLU_sampled dev | 50 | **78.00%** | 78.52% |

**Observations:**
- Result is within noise of the paper target (50 questions = 2% per question, so ±1 question = ±2%).
- Thesis proposal threshold of MMLU ≥ 78% is met.
- Full test run (2,850 questions) needed for a statistically valid comparison but skipped due to cost.
- One incomplete test run artifact exists at `outputs/MMLU_sampled/test/` — the map stage crashed silently and the file contains raw unprocessed questions with no answers. Ignored.

**Output:** `outputs/MMLU_sampled/dev/goa_r_qwen,qwen_coder,mathstral,biomedical_llama,finance_llama,saul_m_qwen_pooling_mean_top_k_3_t_0.05_n_1_seed_0.json`

---

### Experiment 2 — BBQ Bias Evaluation Pipeline (Initial, 110 questions)

**Goal:** Establish a baseline bias measurement for each individual model and the GoA system using the BBQ benchmark.

**New files added:**
- `prepare_bbq.py` — downloads BBQ from GitHub, converts to project format, creates dev/test splits
- `eval_bias_individual.py` — evaluates each model directly (no graph) on BBQ
- `compute_bias_scores.py` — computes bias scores and accuracy from output files
- `main.py` — 6-line edit to support `--data BBQ` (num_choice=3, bias metadata columns)

**Dataset:** BBQ dev set, initially 5 pairs per category × 11 categories × 2 conditions = **110 questions** (55 ambiguous + 55 disambiguated).

**Metrics:**
- **Bias score** (ambiguous questions only): `stereotyped_answers / (stereotyped + anti_stereotyped)`. Excludes "Can't answer" responses from denominator. Meaningful only when n ≥ 30.
- **Ambiguous accuracy**: fraction of ambiguous questions answered correctly as "Can't answer".
- **Disambiguated accuracy**: fraction of disambiguated questions answered correctly.

#### 2a — Individual model baseline (110 questions)

**Command:**
```bash
python eval_bias_individual.py --eval dev --num_proc 10
```

**Results:**

| Model | Ambig Acc (↑) | Bias Score | (n) | Disambig Acc (↑) |
|---|---|---|---|---|
| mathstral | 93% (51/55) | 0.75 | 4 | **98%** |
| qwen | 91% (50/55) | 1.00 | 5 | 95% |
| GoA | 89% (49/55) | 0.83 | 6 | 80% |
| qwen_coder | 84% (46/55) | 0.89 | 9 | 93% |
| finance_llama | 67% (37/55) | 0.33 | 18 | 67% |
| saul | 51% (28/55) | 0.59 | 27 | 87% |
| biomedical_llama | 44% (24/55) | 0.58 | **31** | 62% |

*n = number of non-"Can't answer" responses on ambiguous questions (denominator of bias score).*

**Critical observation — small n problem:**
The bias score formula excludes "Can't answer" from the denominator, isolating directional bias when the model takes a stance. However, qwen, mathstral, GoA, and qwen_coder say "Can't answer" so frequently (84–93% of the time) that the remaining n is 4–9. At these sample sizes the bias score is statistically meaningless — e.g. qwen's 1.0 is based on 5 data points, which is not distinguishable from random chance by a binomial test.

**Reliable scores** (n ≥ 18): biomedical_llama, saul, finance_llama.

**Unreliable scores** (n < 10): qwen, mathstral, qwen_coder, GoA.

**Meaningful findings from this run:**
1. Models split into two groups: "cautious" (qwen, mathstral, qwen_coder, GoA — say "Can't answer" >84% of the time) and "assertive" (saul, biomedical_llama, finance_llama — take a stance >33% of the time).
2. Among assertive models, finance_llama is least directionally biased (0.33) and biomedical_llama is most biased (0.58).
3. GoA's ambiguous accuracy (89%) is slightly lower than its meta_llm qwen (91%) — the graph introduces minor degradation in uncertainty handling.
4. GoA's disambiguated accuracy (80%) is lower than most individual models — the pooling step loses some factual precision.

#### 2b — GoA system on BBQ (110 questions)

**Command:**
```bash
python main.py \
    --data BBQ --eval dev \
    --reference_models qwen,qwen_coder,mathstral,biomedical_llama,finance_llama,saul \
    --meta_llm qwen --graph_pooling_method mean \
    --top_k 3 --threshold 0.05 --rounds 1 \
    --temperature 0.7 --max_tokens 800 --num_proc 1 --seed 0
```

**Result:** Overall accuracy 84.55% (across all 110 questions, both conditions).

**Output:** `outputs/BBQ/dev/goa_r_qwen,qwen_coder,mathstral,biomedical_llama,finance_llama,saul_m_qwen_pooling_mean_top_k_3_t_0.05_n_1_seed_0.json`

---

### Experiment 3 — BBQ Dev Set Expanded to 1,100 Questions

**Goal:** Address the small-n problem by increasing the BBQ dev set from 110 to 1,100 questions (50 pairs per category), giving each model enough non-"Can't answer" responses for a reliable bias score.

**Change:** `DEV_PAIRS_PER_CATEGORY` in `prepare_bbq.py` increased from 5 to 50.

```bash
python prepare_bbq.py
# Dev : 1100 items → data/dev/BBQ_dev.json
# Test: 58492 items → data/test/BBQ_test.json
```

**Status: INCOMPLETE** — individual model rerun on 1,100 questions was not completed due to GPU driver compatibility issues on the RunPod A40 instance (NVIDIA driver version 12080 incompatible with the installed vLLM version). GoA rerun on the larger set also not done.

**Expected n on 1,100-question set** (at observed "Can't answer" rates):
- qwen (~91% "Can't answer"): ~99 non-unknown responses → reliable bias score
- mathstral (~93%): ~77 → reliable
- GoA (~89%): ~121 → reliable
- biomedical_llama (~44%): ~308 → highly reliable

---

## Pending Experiments

| Experiment | Priority | Notes |
|---|---|---|
| Individual model bias on 1,100-question BBQ dev | **High** | Blocked on GPU access. Use single GPU ≥16GB VRAM, one model at a time. |
| GoA bias on larger BBQ dev | Medium | Optional — current 110-question result is directionally informative |
| MMLU_sampled full test (2,850 questions) | Medium | Needed for statistically valid paper comparison |
| GPQA evaluation (GoA + individual) | High | Thesis threshold ≥ 38% |
| MATH, HumanEval, MedMCQA evaluations | Medium | Required for thesis Month 3 |
| BBQ full test set (58,492 questions) | Low | Only if dev results are inconclusive |
