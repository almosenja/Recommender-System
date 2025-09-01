import torch
import numpy as np
import random
import os
import json
from typing import Dict
from config import Config


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(model, checkpoint_path: str, device: str = "cpu"):
    """Load model weights from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    return model


def save_config(config: Config, filepath: str):
    """Save configuration to JSON file."""
    config_dict = config.__dict__
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config(filepath: str) -> Config:
    """Load configuration from JSON file."""
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    return Config(**config_dict)


def compute_user_representations(model, sequences: Dict, device: str = "cuda") -> Dict:
    """Compute user representations from their sequences."""
    model.eval()
    model.to(device)
    user_vecs = {}

    with torch.no_grad():
        for user_id, seq in sequences.items():
            if len(seq) < 1:
                continue

            # Prepare sequence
            seq = seq[-model.max_seq_len:]
            pad_len = model.max_seq_len - len(seq)
            input_seq = torch.tensor([([0] * pad_len + seq)], dtype=torch.long, device=device)

            # Get representation
            hidden = model(input_seq)
            last_hidden = hidden[0, -1, :].cpu().numpy()
            user_vecs[user_id] = last_hidden

    return user_vecs


def build_transfer_matrix(source_vecs: Dict, target_encoder, num_users_target: int) -> torch.Tensor:
    """Build transfer matrix for cross-domain learning."""
    embed_dim = len(next(iter(source_vecs.values())))
    matrix = np.zeros((num_users_target, embed_dim), dtype=np.float32)

    target_user_to_id = {user: i for i, user in enumerate(target_encoder.classes_)}
    hits = 0

    for raw_user, vec in source_vecs.items():
        idx = target_user_to_id.get(raw_user)
        if idx is not None:
            matrix[idx] = vec
            hits += 1

    print(f"Aligned {hits} users to target domain ({hits / num_users_target:.1%} coverage)")
    return torch.from_numpy(matrix)