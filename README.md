# Prompt Optimizer

A research-grade, learnable prompt compression system using Gumbel-Softmax discrete optimization. This tool compresses long system instructions or user prompts into shorter representations while retaining their original semantic meaning, saving API costs and context window usage.

## 🛠️ What We Are Using & For What Tasks

The system relies on a combination of deep learning and natural language heuristics to ensure high compression ratios without losing key details:

### 1. Core Model & Deep Learning
*   **PyTorch & Gumbel-Softmax Selection**: Performs end-to-end differentiable token selection. It decides which tokens to keep or drop based on the target token reduction rate ($\lambda$).
*   **HuggingFace Transformers (DistilBERT)**: Powering the token scoring module (`prompt_optimizer/model/token_scorer.py`). It computes contextual representations of the input prompt to evaluate token importance.

### 2. Semantic Preservation Heuristics (Quality Fixes)
To prevent semantic degradation (such as dropping critical verbs, negation inversion, or losing proper nouns), we integrate:
*   **spaCy (`en_core_web_sm`)**: Used for **Named Entity Recognition (NER)**. Important entities (e.g., proper nouns, places, historical events like `"First World War"`) are detected and receives a logit boost (+8.0) to ensure they are not dropped.
*   **NLTK (Natural Language Toolkit)**: Used for **Part-of-Speech (POS) tagging**. Key content words (nouns, verbs, adjectives, numbers, and question words like `"why"`, `"how"`) are boosted (+5.0 logit), while common auxiliary/copula verbs (e.g., `"is"`, `"was"`, `"does"`) are filtered out to be compressed.
*   **Negation Safeguards**: A critical word dictionary ensures essential negations (e.g., `"not"`, `"no"`, `"never"`, `"without"`, `"don't"`) are always retained, preventing severe meaning inversion (e.g. *"does not use recursion"* $\rightarrow$ *"use recursion"* is avoided).
*   **Contraction & Hyphen Reassembly**: Handles token-level reconstruction to merge separated tokens (e.g. `don ' t` $\rightarrow$ `don't` and `built - in` $\rightarrow$ `built-in`) for clean final outputs.

---

## 🚀 Quick Start

### 1. Install Dependencies
Ensure you have `uv` installed, then synchronize the environment:
```bash
uv sync
```

### 2. Configure Environment
Copy the example environment file and add your `GEMINI_API_KEY`:
```bash
cp .env.example .env
```

---

## 🖥️ Interactive CLI Demo (`scripts/compress.py`)

Run the tool in interactive mode to test prompts manually:
```bash
uv run python scripts/compress.py
```

Or pass direct parameters:
```bash
uv run python scripts/compress.py --prompt "Can you please provide me code for binary search" --task code --lambda 0.1
```

This utility will:
1. Compress the prompt using the Gumbel selector.
2. Estimate cost savings across Gemini, GPT, and Claude.
3. Compare responses side-by-side using the live LLM API.

---

## 🧪 Testing Suites

### 1. Quality Regression Test (`test_quality.py`)
Verifies the optimizer against 55+ hand-crafted regression and adversarial cases (proper nouns, negative constraints, contraction edge cases):
```bash
uv run python test_quality.py
```

### 2. Batch Compression Test (`test_batch.py`)
Runs 102 diverse real-world prompts through the compressor and writes the results to a CSV for manual quality auditing:
```bash
uv run python test_batch.py --lambda 0.1
```
The results are exported to: `results/batch_test_lambda_0_1.csv`

---

## 📊 Key Scientific Results

Our differentiable token selector achieves excellent compression ratios while maintaining high semantic quality:

| Method | Compression Ratio | BERTScore F1 (Quality) | Key Insight |
|:---|:---:|:---:|:---|
| **Identity (Base)** | 0.0% | 1.0000 | Baseline |
| TF-IDF (60%) | 40.0% | 0.8522 | Lacks syntactic structure |
| Sentence Comp. | 32.6% | 0.8573 | Coarse-grained (drops whole sentences) |
| **Ours (Gumbel $\lambda=0.01$)** | **72.3%** | **0.8672** | **Best quality-compression tradeoff** |
| **Ours (Gumbel $\lambda=0.1$)** | **70.4%** | **0.8483** | Highly compressed and coherent |

The **Pareto Frontier** comparison chart mapping quality vs. compression can be viewed at:
[results/pareto_frontier.png](file:///E:/Rakesh/work/Prompt-optimizer/results/pareto_frontier.png)

---

## 📂 Project Structure

```
prompt-optimizer/
├── docs/               # Roadmap, progress tracker, decisions log
├── data/               # Prompt datasets (JSONL)
├── results/            # Auto-generated experiment results & Pareto plot
├── checkpoints/        # Saved model weights
├── prompt_optimizer/   # Main source package
│   ├── llm/            # LLM client + caching
│   ├── evaluation/     # Metrics + eval engine
│   ├── dataset/        # Dataset loading & management
│   ├── baselines/      # Baseline compression methods
│   ├── model/          # Gumbel-Softmax optimizer
│   └── training/       # Training loops
├── scripts/            # Experiment and CLI scripts
├── test_quality.py     # Regression test suite
├── test_batch.py       # Batch CSV test exporter
├── pyproject.toml      # uv project config
└── README.md           # This documentation
```
