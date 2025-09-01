import os
import pandas as pd
from datasets import load_dataset, Features, Value

def load_amazon_reviews(domain: str, save_dir: str = "data", hf_token: str | None = None) -> pd.DataFrame:
    """Load Amazon Reviews dataset for a specific domain and save as CSV."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = f"{save_dir}/amazon_reviews_{domain.lower()}.csv"

    print(f"Downloading Amazon Reviews data for domain {domain}...")
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")

    try:
        ds = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_review_{domain}",
            split="full",
            token=hf_token,
            trust_remote_code=True
        )
    except Exception as e:
        raise RuntimeError(
            "Dataset download failed. Ensure access on Hugging Face and provide a token via "
            "'hf_token' variable or set .env 'HF_TOKEN'."
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

    # Convert to pandas (Arrow zero-copy where possible)
    df = ds.to_pandas()
    df.insert(3, "domain", domain)
    df.to_csv(filepath, index=False)
    print(f"Saved amazon_reviews_{domain}.csv to {save_dir}/")