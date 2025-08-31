# Cross-Domain SASRec
A recommendation system implementing Self-Attentive Sequential Recommendation (SASRec) with cross-domain learning and reinforcement learning capabilities.

## Features
* SASRec Model: Self-attentive sequential recommendation using transformer architecture. ([Original Paper](https://arxiv.org/abs/1808.09781))
* Cross-Domain Transfer: Bridge network for transferring user representations across domains. ([Reference Paper]())
* Cold-Start Handling: Improved performance for users with limited interactions.
* Reinforcement Learning: Policy gradient fine-tuning for improved recommendations.

## Environment
The codes are tested on Python 3.12 with the following libraries:
* Pytorch 2.1.0
* Numpy 1.26.0
* Transformers 4.41.0
* Datasets 3.6.0
* Pandas 2.2.0
* Scikit-learn 1.4.0
* TensorboardX 2.6 (for experiment tracking)
* Tqdm 4.66.0
* Matplotlib 3.8.0

## Installation
```commandline
# Clone the repository
git clone https://github.com/almosenja/Recommender-System.git
cd cross_domain_recsys

# Install dependencies
pip install -r requirements.txt
```

## Basic usage
### 1. Training Mode
Train on a single domain dataset and save metadata:
```commandline
python main.py train \
    --data-path data/amazon_reviews_Electronics.csv \
    --model-name electronics_model \
    --epochs 10 \
    --max-items 5000000 \
    
```

### 2. Transfer Mode
Train with transfer learning from source domain.
```commandline
python main.py transfer \
    --source-data-path data/amazon_reviews_Electronics.csv \
    --target-data-path data/amazon_reviews_Video_Games.csv \
    --epochs 10 \
    --target-max-items 3000000 \
    --source-model-name electronics_model \
    --model-name transfer_model
  ```

### 3. Evaluation Mode
Evaluate transfer model on all, cold-start, and warm-start users.
```commandline
python main.py eval \
    --data-path data/amazon_reviews_Video_Games.csv 
```

### 4. Reinforcement Learning Fine-Tuning
Add RL fine-tuning after cross-domain training:
```commandline
python main.py \
    --source-data data/books.csv \
    --target-data data/movies.csv \
    --epochs 10 \
    --use-rl \
    --rl-epochs 50 \
    --rl-lr 3e-5
````




## Data Preparation
The system expects CSV files with the following columns:
```csv
user,item,timestamp,rating
user_001,item_123,1609459200,5.0
user_001,item_456,1609545600,4.0
```

## Configuration
Key parameters can be adjusted via command-line arguments:

| **Parameter**     | **Default** | **Description**              |
|-------------------|-------------|------------------------------|
| `--hidden-dim`    | 64          | Hidden dimension size        |
| `--num-blocks`    | 2           | Number of transformer blocks |
| `--num-heads`     | 2           | Number of attention heads    |
| `--dropout`       | 0.2         | Dropout rate                 |
| `--max-seq-len`   | 50          | Maximum sequence length      |
| `--batch-size`    | 512         | Batch size for training      |
| `--lr`            | 0.001       | Learning rate                |
| `--epochs`        | 10          | Number of training epochs    |
| `--top-k`         | 10          | K for evaluation metrics     |

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
│  ├─ requirements.txt
├─ notebooks/
│  ├─ 01_EDA.ipynb
│  ├─ 02_recsys_model_development.ipynb
│  └─ 03_recsys_cross_domain_development.ipynb 
└─ README.md
```

## Acknowledgments
* Original SASRec implementation inspired [this work](https://github.com/kang205/SASRec).
* Amazon review datasets from [here](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).
* Cross-domain ideas inspired by [this work](https://arxiv.org/pdf/2110.11154).