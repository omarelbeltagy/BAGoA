 # Graph-of-Agents: A Graph-based Framework for Multi-Agent LLM Collaboration

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) [![ICLR 2026](https://img.shields.io/badge/ICLR'26-red)](https://neurips.cc/)

Official implementation for "Graph-of-Agents: A Graph-based Framework for Multi-Agent LLM Collaboration" accepted by ICLR 2026.  

- Authors: [Sukwon Yun](https://sukwonyun.github.io/), [Jie Peng](https://scholar.google.com/citations?user=wD7PQt0AAAAJ&hl=EN), [Pingzhi Li](https://pingzhili.github.io/), [Wendong Fan](https://openreview.net/profile?id=~Wendong_Fan1), [Jie Chen](https://jiechenjiechen.github.io/), [James Zou](https://www.james-zou.com/), [Guohao Li](https://ghli.org/), and [Tianlong Chen](https://tianlong-chen.github.io/)


## Graph-of-Agents (GoA) Overview
A test-time inference framework that dynamically selects, evaluates, and orchestrates multiple specialized language models as a collaborative graph to solve diverse tasks.

<img src="assets/model.png" width="100%">



## 1. Environment Setup

```bash
conda create -n goa python=3.10 -y
conda activate goa
pip install -r requirements.txt
```

## 2. Serving Models with vLLM

Each model runs as a separate vLLM server on its own GPU. Launch each in a separate terminal (or use `screen`/`tmux`). We recommend serving each model on a different GPU to handle calls efficiently during multiprocessing (we used 6 × A6000 GPUs for experiments with an agent pool of six models):

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --port 8001
CUDA_VISIBLE_DEVICES=2 vllm serve mistralai/Mathstral-7B-v0.1 --port 8002
CUDA_VISIBLE_DEVICES=3 vllm serve ContactDoctor/Bio-Medical-Llama-3-8B --port 8003
CUDA_VISIBLE_DEVICES=4 vllm serve instruction-pretrain/finance-Llama3-8B --port 8004
CUDA_VISIBLE_DEVICES=5 vllm serve Equall/Saul-7B-Instruct-v1 --port 8005
```

Verify a server is running:

```bash
curl http://localhost:8000/v1/models
```

The model endpoints are configured in `endpoint.py`. Update the URLs and ports there if your setup differs.

## RunPod Deployment (6× L40S)

This section covers running GoA on RunPod using 6× NVIDIA L40S GPUs — one model per GPU, matching the original paper setup.

### Pod Configuration

- **Template**: PyTorch (provides torch, transformers, vLLM pre-installed — no conda env needed)
- **GPUs**: 6× L40S (48GB VRAM each)
- **Container disk**: at least 13.5GB
- **Volume disk**: at least 122.6GB (stores downloaded model weights persistently at `/workspace`)
- **Branch**: `l40s`

> **Note**: RunPod's nginx proxy reserves port 8001. The `l40s` branch remaps `qwen_coder` to port 8006.

### Setup Steps

**1. Install missing project dependencies** (PyTorch template already provides torch, vllm, transformers, numpy, requests, loguru):

```bash
pip install -r requirements.txt
```

**2. Set environment variables** (required on every new session before launching servers):

```bash
export HF_HOME=/workspace/hf_cache
export HF_TOKEN=your_huggingface_token
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
```

**3. Launch all 6 model servers** (models download automatically on first run to `/workspace/hf_cache`; load from cache on subsequent runs):

```bash
mkdir -p /workspace/logs /workspace/hf_cache

CUDA_VISIBLE_DEVICES=0 nohup vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 --dtype float16 > /workspace/logs/8000.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup vllm serve Qwen/Qwen2.5-Coder-7B-Instruct \
    --port 8006 --dtype float16 > /workspace/logs/8006.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup vllm serve mistralai/Mathstral-7B-v0.1 \
    --port 8002 --dtype float16 > /workspace/logs/8002.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup vllm serve ContactDoctor/Bio-Medical-Llama-3-8B \
    --port 8003 --dtype float16 > /workspace/logs/8003.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup vllm serve instruction-pretrain/finance-Llama3-8B \
    --port 8004 --dtype float16 > /workspace/logs/8004.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup vllm serve Equall/Saul-7B-Instruct-v1 \
    --port 8005 --dtype float16 > /workspace/logs/8005.log 2>&1 &
```

**4. Verify all servers are ready** (first run: 20–40 min to download; subsequent runs: ~3 min from cache):

```bash
for port in 8000 8006 8002 8003 8004 8005; do
    echo -n "Port $port: "
    curl -s http://localhost:$port/v1/models | python3 -c \
        "import sys,json; d=json.load(sys.stdin); print('OK -', d['data'][0]['id'])" \
        2>/dev/null || echo "NOT READY"
done
```

**5. Clone the project and run:**

```bash
cd /workspace
git clone -b l40s https://github.com/Omar-Beltagui/BAGoA.git
cd BAGoA
python main.py \
    --data MMLU_sampled \
    --eval dev \
    --reference_models qwen,qwen_coder,mathstral,biomedical_llama,finance_llama,saul \
    --meta_llm qwen \
    --graph_pooling_method mean \
    --top_k 3 \
    --seed 0
```

## 3. Running GoA

**Dev run** (small sample for quick testing):

```bash
python main.py \
    --data MMLU_sampled \
    --eval dev \
    --reference_models qwen,qwen_coder,mathstral,biomedical_llama,finance_llama,saul \
    --meta_llm qwen \
    --graph_pooling_method mean \
    --top_k 3 \
    --seed 0
```

**Full evaluation:**

```bash
python main.py \
    --data MMLU_sampled \
    --eval test \
    --reference_models qwen,qwen_coder,mathstral,biomedical_llama,finance_llama,saul \
    --meta_llm qwen \
    --graph_pooling_method mean \
    --top_k 3 \
    --seed 0
```

**Arguments:**

| Argument | Description | Default |
|---|---|---|
| `--data` | Dataset: `GPQA`, `MMLU`, `MMLU_Pro`, `MATH`, `AIME24`, `MedMCQA`, `human_eval` | `GPQA` |
| `--eval` | `dev` (small sample) or `test` (full evaluation) | `test` |
| `--reference_models` | Comma-separated model keys from `endpoint.py` | `qwen,qwen_coder,...` |
| `--meta_llm` | General-purpose model used for node sampling and graph pooling | `qwen` |
| `--graph_pooling_method` | `max`, or `mean`| `mean` |
| `--top_k` | Number of models to select per question | `3` |
| `--threshold` | Minimum edge score to keep a model in the graph | `0.05` |
| `--rounds` | Number of message-passing rounds | `1` |
| `--temperature` | Sampling temperature | `0.7` |
| `--max_tokens` | Max tokens per generation | `800` |
| `--num_proc` | Number of parallel workers | `1` |
| `--seed` | Random seed | `0` |

Results are saved to `outputs/{data}/{eval}/`.

## 4. Adding New Model

### Step 1: Generate a model card

Use `generate_model_card.py` to automatically extract model information from HuggingFace:

```bash
python generate_model_card.py \
    --model_id "meta-llama/Meta-Llama-3-8B-Instruct" \
    --name "llama3" \
    --url "http://localhost:8006/v1/completions" \
    --domain "general" \
    --llm_model "Qwen/Qwen2.5-7B-Instruct" \
    --llm_endpoint "http://localhost:8000/v1/completions"
```

This prints a ready-to-paste dictionary entry.

### Step 2: Add to endpoint.py

Copy the generated entry into `endpoint.py`:

```python
model_endpoint_dict = {
    # ... existing models ...

    "llama3": {
        "url": "http://localhost:8006/v1/completions",
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "max_tokens": 4096,
        "domain": "general",
        "model_card": "- **Domain**: General-purpose\n- **Task Specialization**: ..."
    }
}
```

### Step 3: Serve and run

```bash
# Serve the new model
CUDA_VISIBLE_DEVICES=6 vllm serve meta-llama/Meta-Llama-3-8B-Instruct --port 8006

# Include it in the agent pool
python main.py \
    --data GPQA \
    --eval test \
    --reference_models qwen,qwen_coder,mathstral,biomedical_llama,finance_llama,saul,llama3 \
    --top_k 3
```

## 5. Bias Evaluation (BBQ)

This section covers measuring social bias in individual models and the GoA system using the [BBQ benchmark](https://github.com/nyu-mll/BBQ) (Bias Benchmark for QA). BBQ tests whether models apply social stereotypes when answering ambiguous questions across 11 categories: Age, Disability, Gender, Nationality, Physical Appearance, Race, Religion, SES, and Sexual Orientation.

### How BBQ measures bias

Each BBQ question comes in two versions:
- **Ambiguous**: the context gives no factual basis for an answer — the correct response is always "Can't answer". A model that picks a person based on their demographic group instead is exhibiting bias.
- **Disambiguated**: the context contains a clear factual answer. This measures whether a model can override a stereotype when given evidence.

Two metrics are reported:
- **Bias score** (ambiguous questions): fraction of non-"Can't answer" responses that chose the stereotyped option. Range 0–1; higher = more biased. Only meaningful when the model takes a stance frequently enough (n ≥ 30).
- **Ambiguous accuracy**: fraction of ambiguous questions correctly answered as "Can't answer". Higher = better at recognising uncertainty.
- **Disambiguated accuracy**: fraction of disambiguated questions answered correctly.

### Step 1: Prepare the BBQ dataset

Downloads BBQ from GitHub and creates dev/test splits. The dev set samples 50 ambig+disambig pairs per category (1,100 questions total, 550 ambiguous + 550 disambiguated, balanced across all 11 categories).

```bash
python prepare_bbq.py
# Output: data/dev/BBQ_dev.json  (1,100 questions)
#         data/test/BBQ_test.json (58,492 questions)
```

### Step 2: Evaluate each model individually

Calls each model's endpoint directly — no graph, no pooling. Each model answers all BBQ questions independently, establishing a per-model bias baseline.

Requires one vLLM server running at a time. On a single GPU (≥16GB VRAM), cycle one model at a time:

```bash
# Start one model server, wait for it to respond, then run eval, then stop it
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --dtype float16 &
# wait until: curl http://localhost:8000/v1/models returns a model name
python eval_bias_individual.py --models qwen --eval dev --num_proc 10
pkill -f "vllm serve Qwen/Qwen2.5-7B-Instruct"

# Repeat for each model:
# qwen_coder  → port 8006 → Qwen/Qwen2.5-Coder-7B-Instruct
# mathstral   → port 8002 → mistralai/Mathstral-7B-v0.1
# biomedical_llama → port 8003 → ContactDoctor/Bio-Medical-Llama-3-8B
# finance_llama    → port 8004 → instruction-pretrain/finance-Llama3-8B
# saul             → port 8005 → Equall/Saul-7B-Instruct-v1
```

To run all models at once (requires 6 GPUs as in the RunPod L40S setup):

```bash
python eval_bias_individual.py --eval dev --num_proc 10
```

**Arguments:**

| Argument | Description | Default |
|---|---|---|
| `--data` | Dataset name | `BBQ` |
| `--eval` | `dev` or `test` | `dev` |
| `--models` | Comma-separated model keys to evaluate | all models |
| `--num_proc` | Parallel workers per model | `1` |
| `--seed` | Random seed | `0` |

Output: `outputs/BBQ/dev/individual/{model_name}.json` — one file per model, one entry per question.

### Step 3: Evaluate the GoA system on BBQ

Run the full GoA pipeline on BBQ using the standard `main.py`. Requires all 6 model servers running simultaneously.

```bash
python main.py \
    --data BBQ \
    --eval dev \
    --reference_models qwen,qwen_coder,mathstral,biomedical_llama,finance_llama,saul \
    --meta_llm qwen \
    --graph_pooling_method mean \
    --top_k 3 \
    --seed 0
```

Output: `outputs/BBQ/dev/goa_r_...json`

### Step 4: Compute bias scores

Reads all individual model outputs and the GoA output, then prints a comparison table with overall and per-category scores.

```bash
python compute_bias_scores.py
# Optional: python compute_bias_scores.py --base_dir outputs/BBQ/test
```

Output: printed table + `outputs/BBQ/dev/bias_scores.json`

### Output structure

```
outputs/BBQ/dev/
├── individual/
│   ├── qwen.json
│   ├── qwen_coder.json
│   ├── mathstral.json
│   ├── biomedical_llama.json
│   ├── finance_llama.json
│   └── saul.json
├── goa_r_...json
└── bias_scores.json
```

Each result file contains one entry per question with fields: `question`, `gold_answer`, `answer`, `context_condition`, `category`, `stereotype_ans_idx`, `unknown_ans_idx`. The GoA output additionally includes `sampled_nodes` and `final_response`.

## Project Structure

```
GoA/
├── main.py                  # Clean evaluation script
├── modules.py               # Core GoA pipeline (prompts + graph operations)
├── utils.py                 # Utilities (LLM calls, parsing, evaluation)
├── endpoint.py              # Model endpoint configurations
├── generate_model_card.py   # Tool to generate model cards for new models
├── prepare_bbq.py           # Download and prepare BBQ bias benchmark data
├── eval_bias_individual.py  # Per-model bias evaluation (no GoA pipeline)
├── compute_bias_scores.py   # Compute and compare bias scores across models
├── run.sh                   # Example run script
├── requirements.txt         # Python dependencies
└── data/
    ├── dev/                 # Small dev samples for testing
    └── test/                # Full test sets
```
