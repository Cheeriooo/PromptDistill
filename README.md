# Prompt Optimizer

This is a side project exploring learnable prompt compression using Gumbel-Softmax discrete optimization. The goal of this project is to compress long system instructions or user prompts into shorter representations to save API costs and context window space, while retaining the core semantic meaning.

## System Overview

Unlike simple heuristic compression (like removing random words or sentences), this project implements a learnable, differentiable token selector. 

The optimizer assigns a keep/drop probability to each token using a contextual scorer, uses a Gumbel-Softmax sampler to make a discrete choice, and applies a penalty (lambda) to enforce compression.

### Frameworks & Libraries Used
*   **PyTorch**: Powers the core learnable neural selector and training loops.
*   **Hugging Face Transformers**: Uses `distilbert-base-uncased` as the token scoring model to produce contextual embeddings for each word in the input prompt.
*   **spaCy (en_core_web_sm)**: Used for Named Entity Recognition (NER) to identify proper nouns and prevent them from being dropped.
*   **NLTK**: Used for Part-of-Speech (POS) tagging to detect content words (nouns, verbs, adjectives).
*   **Rich**: Powering the interactive command-line interface.

---

## Core Techniques

To prevent the model from dropping meaning-critical words, the system uses a hybrid approach combining deep learning weights with NLP heuristics:

### 1. Differentiable Token Selection
The model uses the Gumbel-Softmax estimator to approximate discrete keep/drop decisions. This allows the system to train end-to-end based on downstream task rewards and length constraints.

### 2. NER Logit Boosting
Proper nouns (e.g. "First World War", "United States", brand names, dates) are detected via spaCy. A logit boost (+8.0) is applied to these token slots during selection to prevent critical entities from being lost.

### 3. POS Content Boosting
Key parts of speech (nouns, verbs, adjectives, numbers, and question words like "why" or "how") are identified using NLTK and given a logit boost (+5.0). Common auxiliary or copula verbs (like "is", "are", "does") are excluded from this boost to encourage their compression.

### 4. Negation Anchoring
To prevent the model from reversing the meaning of instructions (e.g. converting "do not use recursion" into "use recursion"), a critical word filter locks key negations ("not", "no", "never", "without", "don't") so they are never compressed out.

### 5. Contraction & Hyphen Reassembly
Hugging Face tokenizers split words like "don't" into separate sub-words (`don`, `'`, `t`). A post-processing reassembly step cleans up spacing to output well-formed text (e.g. `don ' t` becomes `don't`, and `built - in` becomes `built-in`).

---

## Test Metrics

These metrics represent the performance of the default model checkpoint across our test suites on a local CPU.

### 1. Batch Test (102 diverse short-to-medium prompts)
*   **Total prompts run**: 102
*   **Compression coefficient (lambda)**: 0.1
*   **Total input tokens**: 1447
*   **Total output tokens**: 1006
*   **Average token reduction**: 30.5%
*   **Average compression latency**: 28.7 ms per prompt
*   **Quality outcome**: 100% semantic preservation across standard test cases (e.g. "What is the capital of india" -> "what capital india").

### 2. Complex Test (8 multi-paragraph, multi-constraint prompts)
*   **Total prompts run**: 8 (average 100+ words per prompt)
*   **Average token reduction**: 37.5%
*   **Average compression latency**: 140.1 ms (cold-start load is ~560 ms, subsequent runs are ~80 ms)
*   **Quality outcome**: Crucial numbered constraints, JSON schemas, code parameters, and negations were completely preserved.

---

## Project Structure

*   `prompt_optimizer/`: Source code package containing the LLM client, evaluation engine, Gumbel-Softmax model, and training scripts.
*   `scripts/compress.py`: Interactive CLI to test custom prompts, see side-by-side completions, and estimate API cost savings.
*   `test_quality.py`: Quality regression test verifying 55+ edge cases (contractions, negations, formatting specs).
*   `test_batch.py`: Exporter script to run 102 diverse prompts and output to a CSV file.
*   `test_complex.py`: Exporter script to run 8 long, multi-paragraph prompts and output to a CSV file.
*   `results/`: Directory containing generated CSV files and comparison plots.

---

## How to Run

### Setup Dependencies
```bash
uv sync
cp .env.example .env
# Open .env and add your GEMINI_API_KEY if you want to test live completions
```

### Interactive CLI Tool
```bash
uv run python scripts/compress.py
```

### Run Batch CSV Test
```bash
uv run python test_batch.py --lambda 0.1
```

### Run Complex Prompt Test
```bash
uv run python test_complex.py
```

### Run Quality Regression Tests
```bash
uv run python test_quality.py
```
