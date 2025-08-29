# Cross-Domain SASRec

This repository implements a **cross-domain recommendation system** that leverages learned representations from a source domain to improve recommendations in a target domain, particularly for cold-start users. 
The system is based on the **SASRec** model with additional cross-domain transfer learning capabilities.

## Features
* SASRec Model: Self-attentive sequential recommendation using transformer architecture. ([Original Paper](https://arxiv.org/abs/1808.09781))
* Cross-Domain Transfer: Bridge network for transferring user representations across domains. ([Reference Paper]())
* Cold-Start Handling: Improved performance for users with limited interactions.
* Reinforcement Learning: RL-based fine-tuning.

## Install
```bash
pip install -r requirements.txt
```

## Quickstart
```bash
# 1) Train source model (defaults: Electronics)
python app.py train-source --source Electronics --max_items 300000

# 2) Transfer train on target (defaults: Video_Games)
python app.py train-target-transfer --source Electronics --target Video_Games --max_items_src 300000 --max_items_tgt 300000

# 3) Train target-only baseline
python app.py train-target-baseline --target Video_Games --max_items_tgt 300000

# 4) Compare on TEST
python app.py eval-compare --target Video_Games --max_items_tgt 300000

# 5) Sample recommendations for a raw user id present in target data
python app.py recommend --target Video_Games --user A3ABCDEFGH --k 10 --use-transfer

# 6) Offline RL demo (tiny)
python app.py rl-demo --target Video_Games --steps 5 --max_items_tgt 300000
```

## Config
Defaults live in `recsys/config.py`. Adjust `ModelCfg`, `TrainCfg`, or domain names. The code reads those values.

## Structure
```
Recommender-System/
├─ cross_domain_recsys/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ utils.py
│  ├─ data.py
│  ├─ datasets.py
│  ├─ models/
│  │  ├─ sasrec.py
│  │  └─ transfer.py
│  ├─ train.py
│  ├─ rl.py
│  └─ recommend.py
├─ 01_EDA.ipynb
├─ 02_recsys_model_development.ipynb
├─ 03_recsys_cross_domain_development.ipynb
├─ app.py
├─ requirements.txt
└─ README.md
```

## Notes
- `item_id==0` is reserved for padding (via `shift_item_id=True`), matching the left-padding in `datasets.py`.
- We use **BCEWithLogitsLoss** on (1 positive + N negatives) logits per instance, flattening tensors to avoid shape bugs.
- For repeatable results, we set seeds and print key metadata.
- `SASRecCD` fuses the last hidden state with source-user vector mapped by a small MLP and a learnable **gate**; users without a source vector fall back to their sequence-only representation.
- The RL demo uses a tiny policy gradient loop as a showcase; treat correct-pick as `"click"` → reward 0.3, else `"skip"` → 0.0. You can plug your own logged events.

## Acknowledgments
* Original SASRec implementation inspired [this work](https://github.com/kang205/SASRec).
* Amazon review datasets from [here](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).
* Cross-domain ideas inspired by [this work](https://arxiv.org/pdf/2110.11154).