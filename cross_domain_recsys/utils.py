import torch
import numpy as np
import random
import os
import json
import pickle
import tqdm
from typing import Dict
from config import Config


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_config(config: Config, filepath: str):
    """Save configuration to JSON file."""
    config_dict = config.__dict__
    with open(filepath, "w") as f:
        json.dump(config_dict, f, indent=2)

def load_model(model, checkpoint_path: str, device: str = "cpu"):
    """Load model weights from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    return model

def load_config(filepath: str) -> Config:
    """Load configuration from JSON file."""
    with open(filepath, "r") as f:
        config_dict = json.load(f)
    return Config(**config_dict)

@torch.no_grad()
def compute_user_representations(model,
                                 sequences,
                                 user_encoder_source,
                                 max_seq_len: int = 50,
                                 batch_size: int = 512,
                                 device: str = "cuda") -> Dict:
    """Compute user representations from their sequences."""
    model.eval().to(device)
    user_vecs = {}

    # Precompute the mapping from encoded ID to raw user string
    raw_users = user_encoder_source.classes_

    # Prepare all sequences and user IDs
    all_seqs = []
    user_ids = []
    for user_id, seq in sequences.items():
        if len(seq) < 1:
            continue
        # Pad and truncate sequences
        seq = seq[-max_seq_len:]
        pad_len = max_seq_len - len(seq)
        padded_seq = [0] * pad_len + seq
        all_seqs.append(padded_seq)
        user_ids.append(user_id)

    # Process in batches
    num_batches = (len(all_seqs) + batch_size - 1) // batch_size
    for i in tqdm(range(num_batches), desc="Computing user representations"):
        start_idx = i * batch_size
        end_idx = min((i+1)*batch_size, len(all_seqs))
        batch_seqs = all_seqs[start_idx:end_idx]
        batch_user_ids = user_ids[start_idx:end_idx]

        # Convert to tensor
        input_seq = torch.tensor(batch_seqs, dtype=torch.long, device=device)

        # Forward pass
        hidden = model(input_seq)
        last_hidden = hidden[:, -1, :].cpu().numpy()

        # Store results using precomputed mapping
        for j, user_id in enumerate(batch_user_ids):
            raw_user = raw_users[user_id]
            user_vecs[raw_user] = last_hidden[j]

    print(f"   Computed user representations for {len(user_vecs)} users and saved to {save_path}.")
    return user_vecs

def load_user_representations(load_path):
    """Load saved user representations."""
    with open(load_path, "rb") as f:
        user_representations = pickle.load(f)
    print(f"   Loaded {len(user_representations)} user representations from {load_path}")
    return user_representations

def build_transfer_matrix(source_vecs: Dict,
                          target_encoder,
                          num_users_target: int) -> torch.Tensor:
    """Build transfer matrix for cross-domain learning."""
    embed_dim = len(next(iter(source_vecs.values())))
    matrix = np.zeros((num_users_target, embed_dim), dtype=np.float32)

    user2idx = {user: i for i, user in enumerate(target_encoder.classes_)}
    hits = 0

    for raw_user, vec in source_vecs.items():
        idx = user2idx.get(raw_user)
        if idx is not None:
            matrix[idx] = vec
            hits += 1

    print(f"   Aligned {hits} users to matrix target space of size {matrix.shape}.")
    return torch.from_numpy(matrix)