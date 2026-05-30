# Prompt Optimizer

A research-grade, learnable prompt compression system using Gumbel-Softmax discrete optimization.

## Docs
- [Roadmap](docs/ROADMAP.md)
- [Progress Tracker](docs/PROGRESS.md)
- [Research Decisions](docs/DECISIONS.md)

## Quick Start

```bash
# Install dependencies
uv sync

# Copy env template
cp .env.example .env
# Fill in your API keys in .env

# Run smoke test (Phase 0 validation)
uv run python smoke_test.py

# Evaluate baselines (Phase 1)
uv run python -m prompt_optimizer.baselines.evaluate_baselines
```

## Project Structure

```
prompt-optimizer/
├── docs/               # Roadmap, progress tracker, decisions log
├── data/               # Prompt datasets (JSONL)
├── results/            # Auto-generated experiment results
├── checkpoints/        # Saved model weights
├── prompt_optimizer/   # Main source package
│   ├── llm/            # LLM client + caching
│   ├── evaluation/     # Metrics + eval engine
│   ├── dataset/        # Dataset loading & management
│   ├── baselines/      # Baseline compression methods
│   ├── model/          # Gumbel-Softmax optimizer (Phase 2+)
│   └── training/       # Training loops (Phase 2+)
├── scripts/            # Experiment runner scripts
├── notebooks/          # Visualization notebooks
├── smoke_test.py       # Phase 0 validation
└── pyproject.toml      # uv project config
```
