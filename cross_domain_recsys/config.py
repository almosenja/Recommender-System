from dataclasses import dataclass

@dataclass
class Paths:
    data_dir: str = "data"
    model_dir: str = "checkpoints"
    tb_dir: str = "runs"

@dataclass
class DataCfg:
    hf_dataset: str = "McAuley-Lab/Amazon-Reviews-2023"
    min_user_interactions: int = 10
    min_item_interactions: int = 10
    max_seq_len: int = 50
    shift_item_id: bool = True  # reserve id 0 for padding

@dataclass
class TrainCfg:
    seed: int = 42
    device: str = "cuda"
    batch_size: int = 512
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-6
    k_eval: int = 10
    neg_samples_train: int = 4
    neg_samples_eval: int = 99

@dataclass
class ModelCfg:
    hidden_dim: int = 64
    num_blocks: int = 2
    num_heads: int = 2
    dropout: float = 0.3
    bridge_hidden: int = 128
    fusion_mode: str = "gate"  # gate|add|concat

@dataclass
class ProjectCfg:
    source_domain: str = "Electronics"
    target_domain: str = "Video_Games"

# convenience global
PATHS = Paths()
DATA = DataCfg()
TRAIN = TrainCfg()
MODEL = ModelCfg()
PROJ  = ProjectCfg()