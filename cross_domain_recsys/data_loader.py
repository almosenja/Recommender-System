import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Optional


class DataProcessor:
    """Handles data loading and preprocessing for recommendation system"""
    def __init__(self, config):
        self.config = config

    def load_csv(self, filepath: str, max_items: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
        """Load data from CSV file with column mapping."""
        print(f"   Loading data from {filepath}...")
        df = pd.read_csv(filepath)

        # Ensure required columns exist
        required_cols = ["user", "item", "timestamp"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe after renaming")

        # Convert types
        df["user"] = df["user"].astype(str)
        df["item"] = df["item"].astype(str)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

        # Random sampling if max_items specified
        if max_items and len(df) > max_items:
            frac = max_items / len(df)
            df = df.sample(frac=frac, random_state=seed).reset_index(drop=True)

        print(f"   Loaded {len(df)} rows with {df['user'].nunique()} users and {df['item'].nunique()} items")
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter users and items based on minimum interaction thresholds"""
        df["label"] = 1.0

        while True:
            initial_rows = df.shape[0]

            user_counts = df.groupby("user").size()
            valid_users = user_counts[user_counts >= self.config.min_user_interactions].index
            df = df[df["user"].isin(valid_users)]

            item_counts = df.groupby("item").size()
            valid_items = item_counts[item_counts >= self.config.min_item_interactions].index
            df = df[df["item"].isin(valid_items)]

            if df.shape[0] == initial_rows:
                break

        print(f"   After filtering: {len(df)} rows, {df['user'].nunique()} users, "
              f"{df['item'].nunique()} items")

        return df

    def encode_ids(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
        """Encode user and item IDs to sequential integers"""
        df_encoded = df.copy()

        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()

        df_encoded["user_id"] = user_encoder.fit_transform(df_encoded["user"])
        df_encoded["item_id"] = item_encoder.fit_transform(df_encoded["item"])
        df_encoded["item_id"] = df_encoded["item_id"] + 1  # Reserve 0 for padding

        return df_encoded, user_encoder, item_encoder

    def create_sequences(self, df: pd.DataFrame) -> Dict[int, list]:
        """Create user interaction sequences sorted by timestamp"""
        df_sorted = df.sort_values(["user_id", "timestamp"])
        user_sequences = {}

        for uid, group in df_sorted.groupby("user_id"):
            items = group["item_id"].tolist()
            user_sequences[uid] = items

        seq_lens = [len(seq) for seq in user_sequences.values()]
        print(f"   Created {len(user_sequences)} sequences")
        print(f"   Sequence length - Min: {min(seq_lens)}, Max: {max(seq_lens)}, Avg: {np.mean(seq_lens):.1f}")

        return user_sequences

    def split_sequences(self, user_sequences: Dict[int, list]) -> Tuple[dict, dict, dict]:
        """
        Split sequences into train/val/test using leave-one-out strategy.
        Returns: train_sequences, val_sequences, test_sequences
        """
        train_seqs = {}
        val_data = {}
        test_data = {}

        for user, seq in user_sequences.items():
            if len(seq) < 3:  # Need at least 3 items
                continue

            train_seqs[user] = seq[:-2]
            val_data[user] = (seq[:-2], seq[-2])  # Predict second-to-last
            test_data[user] = (seq[:-1], seq[-1])  # Predict last

        print(f"   Split complete - Train: {len(train_seqs)}, Val: {len(val_data)}, Test: {len(test_data)}")

        return train_seqs, val_data, test_data