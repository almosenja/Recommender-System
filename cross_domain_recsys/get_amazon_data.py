import os
from datasets import load_dataset, Features, Value

def load_review_data(domain, token=None, save_dir="data"):
    os.makedirs(save_dir, exist_ok=True)
    filepath = f"{save_dir}/amazon_reviews_{domain}.csv"

    if not os.path.exists(filepath):
        print(f"File {filepath} not found. Downloading dataset for domain '{domain}'...")
        ds = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_review_{domain}",
            split="full",
            trust_remote_code=True,
            token=token
        )

        # Keep only needed columns
        ds = ds.select_columns(["user_id", "parent_asin", "rating", "timestamp"])
        ds = ds.rename_columns({"user_id": "user", "parent_asin": "item"})
        ds = ds.cast(Features({
            "user": Value("string"),
            "item": Value("string"),
            "rating": Value("float32"),
            "timestamp": Value("int64"),
        }))

        # Convert to pandas
        df = ds.to_pandas()
        df.insert(3, "domain", domain)
        df.to_csv(f"{save_dir}/amazon_reviews_{domain}.csv", index=False)
        print(f"Saved review data to {filepath}")

    else:
        print(f"File {filepath} already exists.")


def load_meta_data(domain, token=None, save_dir="data"):
    os.makedirs(save_dir, exist_ok=True)
    filepath = f"{save_dir}/amazon_meta_{domain}.csv"

    if not os.path.exists(filepath):
        print(f"File {filepath} not found. Downloading metadata dataset for domain '{domain}'...")
        ds = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{domain}",
            split="full",
            trust_remote_code=True,
            token=token
        )

        # Keep only needed columns
        ds = ds.select_columns(["title", "parent_asin"])
        ds = ds.rename_columns({"parent_asin": "item"})
        ds = ds.cast(Features({
            "item": Value("string"),
            "title": Value("string")
        }))

        # Convert to pandas
        df = ds.to_pandas()
        df.to_csv(f"{save_dir}/amazon_meta_{domain}.csv", index=False)
        print(f"Saved meta data to {filepath}")

    else:
        print(f"File {filepath} already exists.")