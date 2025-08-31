import os
from dataclasses import dataclass


@dataclass
class Config:
    # Data settings
    data_path: str = ""
    save_dir: str = "outputs"
    model_dir: str = "models"

    # Data preprocessing
    min_user_interactions: int = 10
    min_item_interactions: int = 10
    max_seq_len: int = 50

    # Model architecture
    hidden_dim: int = 64
    num_blocks: int = 2
    num_heads: int = 2
    dropout: float = 0.4

    # Training settings
    batch_size: int = 512
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    neg_samples_train: int = 4
    neg_samples_eval: int = 99

    # Transfer learning
    source_domain: str = ""
    target_domain: str = ""
    bridge_hidden: int = 128

    # RL settings
    rl_epochs: int = 50
    rl_lr: float = 3e-5
    entropy_coeff: float = 0.015
    temperature: float = 1.0
    baseline_momentum: float = 0.9

    # Evaluation
    top_k: int = 10

    # General
    seed: int = 42
    device: str = "cuda"

    def __post_init__(self):
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)