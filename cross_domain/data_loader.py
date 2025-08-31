import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple, Optional


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.user_encoder = None
        self.item_encoder = None

    def load_data(self, filepath: str, columns_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """Load data from CSV with flexible column mapping."""
        df = pd.read_csv(filepath)

        # Default mapping for common datasets
        default_mappings = {
            "amazon": {
                "user_id": "user",
                "parent_asin": "item",
                "rating": "rating",
                "timestamp": "timestamp"},
            # etc.
        }

        # Apply column mapping
        if columns_mapping:
            df = df.rename(columns=columns_mapping)
        else:
            # Try to auto-detect format
            for dataset_type, mapping in default_mappings.items():
                if all(col in df.columns for col in mapping.keys()):
                    df = df.rename(columns=mapping)
                    print(f"Auto-detected {dataset_type} format")
                    break

        # Ensure required columns exist
        required_cols = ["user", "item", "rating", "timestamp"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Need: {required_cols}, Got: {df.columns.tolist()}")

        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter interactions and create implicit feedback."""
        df["label"] = 1.0

        # Filter by minimum interactions
        user_counts = df.groupby("user").size()
        valid_users = user_counts[user_counts >= self.config.min_user_interactions].index

        item_counts = df.groupby("item").size()
        valid_items = item_counts[item_counts >= self.config.min_item_interactions].index

        df_filtered = df[df["user"].isin(valid_users) & df["item"].isin(valid_items)]

        print(f"After filtering: {len(df_filtered)} rows, "
              f"{df_filtered['user'].nunique()} users, "
              f"{df_filtered['item'].nunique()} items")

        return df_filtered

    def encode_ids(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
        """Encode user and item IDs."""
        df_encoded = df.copy()

        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

        df_encoded["user_id"] = self.user_encoder.fit_transform(df_encoded["user"])
        df_encoded["item_id"] = self.item_encoder.fit_transform(df_encoded["item"])
        df_encoded["item_id"] = df_encoded["item_id"] + 1  # Reserve 0 for padding

        return df_encoded, self.user_encoder, self.item_encoder

    def create_sequences(self, df: pd.DataFrame) -> Dict[int, list]:
        """Create user interaction sequences."""
        df_sorted = df.sort_values(["user_id", "timestamp"])
        user_sequences = {}

        for uid, group in df_sorted.groupby("user_id"):
            items = group["item_id"].tolist()
            user_sequences[uid] = items

        return user_sequences

    def split_sequences(self, user_sequences: Dict[int, list]) -> Tuple[Dict, Dict, Dict]:
        """Leave-one-out split for train/val/test."""
        train_seqs = {}
        val_data = {}
        test_data = {}

        for user, seq in user_sequences.items():
            if len(seq) < 3:
                continue

            train_seqs[user] = seq[:-2]
            val_data[user] = (seq[:-2], seq[-2])
            test_data[user] = (seq[:-1], seq[-1])

        return train_seqs, val_data, test_data


class RecDataset(Dataset):
    def __init__(self, data, num_items, max_seq_len=50,
                 pos_items_by_user=None, mode="train", neg_samples=1):
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.neg_samples = neg_samples
        self.all_pos = pos_items_by_user

        self.samples = []
        if mode == "train":
            for user, seq in data.items():
                for i in range(1, len(seq)):
                    self.samples.append({
                        "user": user,
                        "input_seq": seq[:i],
                        "target": seq[i],
                        "full_seq": seq
                    })
        else:
            for user, (seq, target) in data.items():
                self.samples.append({
                    "user": user,
                    "input_seq": seq,
                    "target": target,
                    "full_seq": seq + [target]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        user = sample["user"]
        seq = sample["input_seq"]
        target = sample["target"]

        # Truncate if needed
        if len(seq) > self.max_seq_len:
            seq = seq[-self.max_seq_len:]

        # Left-pad with zeros
        pad_len = self.max_seq_len - len(seq)
        padded_seq = [0] * pad_len + seq

        # Negative sampling
        forbid = self.all_pos[user] if self.all_pos else set(sample["full_seq"])
        neg_items = set()

        while len(neg_items) < self.neg_samples:
            neg = np.random.randint(1, self.num_items)
            if neg not in forbid:
                neg_items.add(neg)

        return {
            "user": sample["user"],
            "input_seq": torch.tensor(padded_seq, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.long),
            "neg_items": torch.tensor(list(neg_items), dtype=torch.long)
        }