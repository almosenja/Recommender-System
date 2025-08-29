import os, random, numpy as np, torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(pref: str = "cuda"):
    if pref == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)