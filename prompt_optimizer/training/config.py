"""
Training hyperparameter configuration.

All training settings live here as a typed Pydantic model.
Values are read from environment variables with sane defaults.

Usage:
    from prompt_optimizer.training.config import TrainingConfig

    cfg = TrainingConfig()          # loads from .env
    cfg = TrainingConfig(lambda_=0.5, num_epochs=30)  # override
    print(cfg.model_dump())
"""

from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

load_dotenv()


class TrainingConfig(BaseModel):
    """All hyperparameters for the Gumbel-Softmax prompt optimizer."""

    model_config = {"protected_namespaces": ()}

    # ------------------------------------------------------------------
    # Encoder / backbone
    # ------------------------------------------------------------------
    scorer_base_model: str = Field(
        default_factory=lambda: os.getenv("SCORER_BASE_MODEL", "distilbert-base-uncased"),
        description="HuggingFace model used for token embeddings (frozen during training).",
    )
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        description="Sentence-transformer model for proxy reward computation.",
    )

    # ------------------------------------------------------------------
    # Task conditioning
    # ------------------------------------------------------------------
    task_types: list[str] = Field(
        default=["qa", "summarization", "code", "creative", "general"],
        description="All task types. Order determines embedding indices.",
    )
    task_embed_dim: int = Field(
        default=32,
        description="Dimensionality of the learned task-type embedding.",
    )

    # ------------------------------------------------------------------
    # Token scorer MLP
    # ------------------------------------------------------------------
    scorer_hidden_dim: int = Field(
        default=256,
        description="Hidden layer size in the token importance MLP.",
    )
    scorer_dropout: float = Field(
        default=0.1,
        ge=0.0,
        lt=1.0,
        description="Dropout rate in the scorer MLP.",
    )

    # ------------------------------------------------------------------
    # Gumbel-Softmax
    # ------------------------------------------------------------------
    gumbel_tau_start: float = Field(
        default=2.0,
        description="Initial Gumbel-Softmax temperature (high = soft/exploratory).",
    )
    gumbel_tau_end: float = Field(
        default=0.1,
        description="Final temperature after annealing (low = near-discrete).",
    )
    gumbel_tau_anneal_steps: int = Field(
        default=500,
        description="Number of steps over which to anneal temperature.",
    )

    # ------------------------------------------------------------------
    # Loss function
    # ------------------------------------------------------------------
    lambda_: float = Field(
        default_factory=lambda: float(os.getenv("LAMBDA", "0.1")),
        alias="lambda_coeff",
        description=(
            "Compression penalty coefficient. "
            "Higher = more aggressive compression, lower quality. "
            "Sweep: [0.01, 0.05, 0.1, 0.5] to build Pareto curve."
        ),
    )

    model_config = {"populate_by_name": True}

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    learning_rate: float = Field(
        default_factory=lambda: float(os.getenv("LEARNING_RATE", "5e-4")),
        description="AdamW learning rate.",
    )
    weight_decay: float = Field(
        default=1e-4,
        description="L2 regularization weight decay.",
    )
    batch_size: int = Field(
        default_factory=lambda: int(os.getenv("BATCH_SIZE", "8")),
        description="Training batch size.",
    )
    num_epochs: int = Field(
        default_factory=lambda: int(os.getenv("NUM_EPOCHS", "20")),
        description="Number of full passes through the training dataset.",
    )
    warmup_steps: int = Field(
        default_factory=lambda: int(os.getenv("WARMUP_STEPS", "100")),
        description="Linear LR warmup steps at the start of training.",
    )
    grad_clip: float = Field(
        default=1.0,
        description="Max gradient norm for clipping (prevents exploding gradients).",
    )

    # ------------------------------------------------------------------
    # Evaluation during training
    # ------------------------------------------------------------------
    eval_every_n_steps: int = Field(
        default=50,
        description="Run proxy evaluation (no LLM) every N training steps.",
    )
    llm_eval_every_n_steps: int = Field(
        default=200,
        description="Run true LLM evaluation (with Gemini) every N steps. Expensive!",
    )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    checkpoint_dir: str = Field(
        default="checkpoints",
        description="Directory to save model checkpoints.",
    )
    save_every_n_epochs: int = Field(
        default=5,
        description="Save a checkpoint every N epochs.",
    )

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    seed: int = Field(
        default=42,
        description="Random seed for reproducible training.",
    )
    device: str = Field(
        default="auto",
        description="'auto' | 'cpu' | 'cuda' — auto-detects GPU if available.",
    )

    @field_validator("device", mode="before")
    @classmethod
    def resolve_device(cls, v: str) -> str:
        """Resolve 'auto' to 'cuda' or 'cpu' at config construction time."""
        if v == "auto":
            try:
                import torch  # noqa: PLC0415
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return v

    def model_post_init(self, __context) -> None:
        """Final resolution pass — ensures device is never 'auto'."""
        if self.device == "auto":
            try:
                import torch  # noqa: PLC0415
                object.__setattr__(self, "device", "cuda" if torch.cuda.is_available() else "cpu")
            except ImportError:
                object.__setattr__(self, "device", "cpu")

    def task_to_idx(self, task: str) -> int:
        """Convert a task type string to its embedding index."""
        try:
            return self.task_types.index(task)
        except ValueError:
            return self.task_types.index("general")

    def summary(self) -> str:
        """One-line summary for logging."""
        return (
            f"λ={self.lambda_:.3f} | lr={self.learning_rate} | "
            f"epochs={self.num_epochs} | batch={self.batch_size} | "
            f"device={self.device} | tau={self.gumbel_tau_start}→{self.gumbel_tau_end}"
        )
