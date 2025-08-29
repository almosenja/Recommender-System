import os
import pandas as pd
from datasets import load_dataset, Features, Value
from sklearn.preprocessing import LabelEncoder

from .config import PATHS, DATA

def load_amazon_reviews(domain: str, save_dir: str = None, max_items=None, seed=42):
    save_dir = save_dir or PATHS.data_dir
    os.makedirs(save_dir, exist_ok=True)
    filepath = f"{save_dir}/amazon_reviews_{domain}.csv"

    if not os.path.exists(filepath):
        print(f"[data] downloading {domain} → {filepath}")
        ds = load_dataset(DATA.hf_dataset, f"raw_review_{domain}", split="full", trust_remote_code=True)
        ds = ds.select_columns(["user_id", "parent_asin", "rating", "timestamp"])
        ds = ds.rename_columns({"user_id": "user", "parent_asin": "item"})
        ds = ds.cast(Features({
            "user": Value("string"),
            "item": Value("string"),
            "rating": Value("float32"),
            "timestamp": Value("int64"),
        }))
        df = ds.to_pandas()
        df.insert(3, "domain", domain)
        df.to_csv(filepath, index=False)

    df = pd.read_csv(filepath)
    if max_items is not None and len(df) > max_items:
        df = df.sample(n=max_items, random_state=seed).reset_index(drop=True)
    print(f"[data] loaded {domain}: {len(df)} rows")
    return df

def preprocess_dataset(df, min_user_interactions=None, min_item_interactions=None):
    min_user_interactions = min_user_interactions or DATA.min_user_interactions
    min_item_interactions = min_item_interactions or DATA.min_item_interactions

    df = df.copy()
    df["label"] = 1.0  # implicit feedback
    user_counts = df.groupby("user").size()
    valid_users = user_counts[user_counts >= min_user_interactions].index
    item_counts = df.groupby("item").size()
    valid_items = item_counts[item_counts >= min_item_interactions].index
    df_f = df[df["user"].isin(valid_users) & df["item"].isin(valid_items)]
    print(f"[data] after filter: {len(df_f)} rows, {df_f['user'].nunique()} users, {df_f['item'].nunique()} items")
    return df_f

def label_encoder(df, shift_item_id: bool = DATA.shift_item_id):
    df = df.copy()
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    domain_enc = LabelEncoder()

    df["user_id"] = user_enc.fit_transform(df["user"])
    df["item_id"] = item_enc.fit_transform(df["item"])
    if shift_item_id:
        df["item_id"] = df["item_id"] + 1  # reserve 0 for padding
    df["domain_id"] = domain_enc.fit_transform(df["domain"])
    return df, user_enc, item_enc, domain_enc

def create_user_sequences(df):
    df_sorted = df.sort_values(["user_id", "timestamp"])
    user_sequences = {}
    for uid, group in df_sorted.groupby("user_id"):
        items = group["item_id"].tolist()
        user_sequences[uid] = items
    print(f"[seq] users={len(user_sequences)}  maxL={max(len(v) for v in user_sequences.values())}  minL={min(len(v) for v in user_sequences.values())}")
    return user_sequences

def sequences_loo_split(user_sequences):
    train_seqs = {}
    val_data = {}
    test_data = {}
    for user, seq in user_sequences.items():
        if len(seq) < 3:
            continue
        train_seqs[user] = seq[:-2]
        val_data[user] = (seq[:-2], seq[-2])
        test_data[user] = (seq[:-1], seq[-1])
    print(f"[split] train_users={len(train_seqs)}  val_users={len(val_data)}  test_users={len(test_data)}")
    return train_seqs, val_data, test_data

def split_cold_warm(train_sequences_tgt, cold_threshold=1):
    cold_users = {u for u, seq in train_sequences_tgt.items() if len(seq) <= cold_threshold}
    warm_users = {u for u, seq in train_sequences_tgt.items() if len(seq) >= (cold_threshold + 1)}
    return cold_users, warm_users

def filter_split(split_dict, keep_users):
    return {u: v for u, v in split_dict.items() if u in keep_users}

def analyze_user_overlap(df_source, df_target):
    users_src = set(df_source["user"].unique())
    users_tgt = set(df_target["user"].unique())
    common_users = users_src.intersection(users_tgt)
    print(f"[overlap] source={len(users_src)} target={len(users_tgt)} common={len(common_users)} "
          f"tgt_in_src={len(common_users)/max(1,len(users_tgt))*100:.2f}% src_in_tgt={len(common_users)/max(1,len(users_src))*100:.2f}%")
    return users_src, users_tgt, common_users