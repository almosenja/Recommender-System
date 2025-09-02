# Cross-Domain SASRec
A recommendation system implementing Self-Attentive Sequential Recommendation (SASRec) with cross-domain learning and reinforcement learning capabilities.

## Features
* SASRec Model: Self-attentive sequential recommendation using transformer architecture.
* Cross-Domain Transfer: Bridge network for transferring user representations across domains.
* Cold-Start Handling: Improved performance for users with limited interactions.
* Reinforcement Learning: Policy gradient offline fine-tuning for improved recommendations.

## Environment
The codes are tested on Python 3.12 using PyTorch environment.

## Installation
```commandline
# Clone the repository
git clone https://github.com/almosenja/Recommender-System.git
cd cross_domain_recsys

# Install dependencies
pip install -r requirements.txt
```

## Basic usage
### 1. Train SASRec on the source domain (e.g., Books):
```commandline
python main.py train \
    --data-path data/amazon_reviews_Books.csv \
    --model-name books_model \
    --epochs 10 \
    --max-items 5000000
```

### 2. Transfer on the target domain (e.g., Movies & TV):
```commandline
python main.py transfer \
    --source-data-path data/amazon_reviews_Books.csv \
    --target-data-path data/amazon_reviews_Movies_and_TV.csv \
    --epochs 10 \
    --target-max-items 5000000 \
    --source-model-name books_model \
    --model-name transfer_model
  ```

### 3. Evaluate (overall, cold-start, warm-start) on target domain using transfer model:
```commandline
python main.py eval \
    --data-path data/amazon_reviews_Movies_and_TV.csv \
    --model-name transfer_model \
    --top-k 10
```

### 4. Reinforcement Fine-Tune offline on transfer model:
```commandline
python main.py rl_finetune \
    --data-path data/amazon_reviews_Movies_and_TV.csv \
    --rl-epochs 10 \
    --model-name transfer_model
    --rl-lr 1e-4
````

### 5. Inference with optional metadata:
```commandline
python main.py inference \
    --data-path data/amazon_reviews_Movies_and_TV.csv \
    --model-name transfer_model \
    --num-samples 5 \
    --top-k 10 \
    --metadata-path data/amazon_meta_Movies_and_TV.csv
```

## Data Preparation
The system expects CSV files with the following columns:
```csv
user,item,timestamp,rating
user_001,item_123,1609459200,5.0
user_001,item_456,1609545600,4.0
```

Download Amazon Reviews 2023 datasets from [Hugging Face](https://huggingface.co/datasets/amazon_reviews_multi) or use the provided script:
```commandline
python main.py get_amazon_data \
    --review-or-meta review \
    --download-dir data/ \
    --domains Books Movies_and_TV etc..
```


## CLI Arguments
CLI arguments are categorized as follows:
* Global: `--device`,` --seed`, `--save-dir`, `--model-dir`, `--model-name`
* Data: `--data-path`, `--max-items`, `--max-seq-len`
* Model: `--hidden-dim`, `--num-blocks`, `--num-heads`, `--dropout`, `--epochs`, `--batch-size`, `--lr`, `--weight-decay`
* Transfer: `--source-data-path`, `--target-data-path`, `--source-model-name`, `--bridge-hidden`
* RL: `--rl-epochs`, `--rl-lr`, `--entropy-coeff`
* Inference: `--metadata-path`, `--item-col`, `--title-col`, `--user-id`, `--num-samples`, `--top-k`
* Amazon Download: `--review-or-meta {review,meta}`, `--download-dir`, `--domains`

## Project Structure
```
Recommender-System/
├─ cross_domain_recsys/
│  ├─ main.py
│  ├─ config.py 
│  ├─ data_loader.py 
│  ├─ dataset.py 
│  ├─ models.py     
│  ├─ train.py
│  ├─ evaluate.py  
│  ├─ recommend.py        
│  ├─ rl_trainer.py 
│  ├─ utils.py
│  ├─ visualization.py
│  ├─ get_amazon_data.py
│  ├─ inference.py
│  ├─ requirements.txt
├─ notebooks/
│  ├─ 01_EDA.ipynb
│  ├─ 02_recsys_model_development_fin.ipynb
│  └─ 03_recsys_cross_domain_development_fin.ipynb 
└─ README.md
```

## Acknowledgments
* **Matrix Factorization**: Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems.
* **NeuMF**: He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017). Neural Collaborative Filtering.
* **SASRec**: Kang, W.-C., & McAuley, J. (2018). Self-Attentive Sequential Recommendation.
* **Reinforcement Learning**: Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning.