import argparse
import os
import torch
import json
import pandas as pd

# Import all modules
from config import Config
from data_loader import DataProcessor, RecDataset
from models import SASRec, SASRecTransfer
from trainers import Trainer, TransferTrainer
from evaluators import Evaluator
from rl_trainer import RLTrainer
from inference import RecommendationInference
from utils import set_seed, load_model, save_config, load_config, compute_user_representations, build_transfer_matrix
from torch.utils.data import DataLoader


def train_mode(args):
    """Train a new model from scratch."""
    print("=" * 80)
    print("TRAINING MODE")
    print("=" * 80)

    # Setup config
    config = Config(
        data_path=args.data_path,
        save_dir=args.save_dir,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        device=args.device,
        seed=args.seed
    )

    set_seed(config.seed)
    save_config(config, os.path.join(config.save_dir, "config.json"))

    # Load and process data
    print("\n1. Loading data...")
    processor = DataProcessor(config)

    # Handle column mapping if provided
    col_mapping = None
    if args.column_mapping:
        col_mapping = json.loads(args.column_mapping)

    df = processor.load_data(args.data_path, col_mapping)
    df_filtered = processor.preprocess(df)
    df_encoded, user_enc, item_enc = processor.encode_ids(df_filtered)

    # Create sequences
    print("\n2. Creating sequences...")
    user_sequences = processor.create_sequences(df_encoded)
    train_seqs, val_seqs, test_seqs = processor.split_sequences(user_sequences)

    # Save encoders
    import pickle
    with open(os.path.join(config.save_dir, "encoders.pkl"), "wb") as f:
        pickle.dump({"user": user_enc, "item": item_enc}, f)

    # Create datasets
    print("\n3. Creating datasets...")
    num_items = df_encoded["item_id"].max() + 1
    pos_items_by_user = {u: set(seq) for u, seq in user_sequences.items()}

    train_dataset = RecDataset(train_seqs, num_items, config.max_seq_len,
                               pos_items_by_user, mode="train", neg_samples=config.neg_samples_train)
    val_dataset = RecDataset(val_seqs, num_items, config.max_seq_len,
                             pos_items_by_user, mode="val", neg_samples=config.neg_samples_eval)
    test_dataset = RecDataset(test_seqs, num_items, config.max_seq_len,
                              pos_items_by_user, mode="test", neg_samples=config.neg_samples_eval)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    # Create and train model
    print("\n4. Training model...")
    model = SASRec(
        num_items=num_items,
        hidden_dim=config.hidden_dim,
        max_seq_len=config.max_seq_len,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        dropout=config.dropout
    )

    trainer = Trainer(config)
    best_ndcg = trainer.train(model, train_loader, val_loader, config.epochs, args.model_name)

    # Final evaluation
    print("\n5. Final evaluation on test set...")
    evaluator = Evaluator(config)
    model = load_model(model, os.path.join(config.model_dir, f"{args.model_name}.pth"), config.device)
    test_metrics = evaluator.evaluate(model, test_loader)

    print(f"\nTest Results:")
    print(f"  HR@{config.top_k}: {test_metrics['HR@K']:.4f}")
    print(f"  NDCG@{config.top_k}: {test_metrics['NDCG@K']:.4f}")
    print(f"  MRR@{config.top_k}: {test_metrics['MRR@K']:.4f}")

    # Save test results
    with open(os.path.join(config.save_dir, "test_results.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)


def transfer_mode(args):
    """Train with transfer learning from source domain."""
    print("=" * 80)
    print("TRANSFER LEARNING MODE")
    print("=" * 80)

    # Load source model config
    source_config = load_config(os.path.join(args.source_model_dir, "config.json"))

    # Setup target config
    config = Config(
        data_path=args.data_path,
        save_dir=args.save_dir,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        hidden_dim=source_config.hidden_dim,  # Must match source
        bridge_hidden=args.bridge_hidden,
        device=args.device,
        seed=args.seed
    )

    set_seed(config.seed)

    # Load source model and data
    print("\n1. Loading source model...")
    import pickle
    with open(os.path.join(args.source_model_dir, "encoders.pkl"), "rb") as f:
        source_encoders = pickle.load(f)

    # Create dummy source model to load weights
    source_model = SASRec(
        num_items=100000,  # Will be overridden
        hidden_dim=source_config.hidden_dim,
        max_seq_len=source_config.max_seq_len,
        num_blocks=source_config.num_blocks,
        num_heads=source_config.num_heads,
        dropout=source_config.dropout
    )

    source_model = load_model(
        source_model,
        os.path.join(args.source_model_dir, f"{args.source_model_name}.pth"),
        config.device
    )

    # Load and process target data
    print("\n2. Loading target data...")
    processor = DataProcessor(config)

    col_mapping = None
    if args.column_mapping:
        col_mapping = json.loads(args.column_mapping)

    df = processor.load_data(args.data_path, col_mapping)
    df_filtered = processor.preprocess(df)
    df_encoded, user_enc, item_enc = processor.encode_ids(df_filtered)

    # Create sequences
    user_sequences = processor.create_sequences(df_encoded)
    train_seqs, val_seqs, test_seqs = processor.split_sequences(user_sequences)

    # Build transfer matrix
    print("\n3. Building transfer matrix...")
    # First, load source sequences (you'd need to save these during source training)
    # For now, we'll compute from source model
    source_user_vecs = {}  # In practice, load these from source training

    num_users_target = df_encoded["user_id"].max() + 1
    num_items_target = df_encoded["item_id"].max() + 1

    # Create transfer matrix (simplified - in practice, you'd compute from source sequences)
    transfer_matrix = build_transfer_matrix(
        source_user_vecs,
        processor.user_encoder,
        num_users_target
    )

    # Create datasets with transfer
    class TransferDataset(RecDataset):
        def __init__(self, *args, transfer_mat=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.transfer_mat = transfer_mat

        def __getitem__(self, idx):
            out = super().__getitem__(idx)
            if self.transfer_mat is not None:
                out["transfer_src"] = self.transfer_mat[out["user"]].float()
            return out

    train_dataset = TransferDataset(train_seqs, num_items_target, config.max_seq_len,
                                    transfer_mat=transfer_matrix, mode="train", neg_samples=4)
    val_dataset = TransferDataset(val_seqs, num_items_target, config.max_seq_len,
                                  transfer_mat=transfer_matrix, mode="val", neg_samples=99)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Create transfer model
    print("\n4. Training transfer model...")
    target_base = SASRec(
        num_items=num_items_target,
        hidden_dim=source_config.hidden_dim,
        max_seq_len=config.max_seq_len,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        dropout=config.dropout
    )

    model = SASRecTransfer(
        target_base,
        hidden_dim=source_config.hidden_dim,
        bridge_hidden=config.bridge_hidden,
        dropout=config.dropout
    )

    trainer = TransferTrainer(config)
    trainer.train(model, train_loader, val_loader, config.epochs, args.model_name)


def eval_mode(args):
    """Evaluate a trained model."""
    print("=" * 80)
    print("EVALUATION MODE")
    print("=" * 80)

    # Load config
    config = load_config(os.path.join(args.model_dir, "config.json"))
    config.device = args.device
    set_seed(config.seed)

    # Load data
    print("\n1. Loading data...")
    processor = DataProcessor(config)

    col_mapping = None
    if args.column_mapping:
        col_mapping = json.loads(args.column_mapping)

    df = processor.load_data(args.data_path, col_mapping)
    df_filtered = processor.preprocess(df)
    df_encoded, _, _ = processor.encode_ids(df_filtered)

    # Load encoders
    import pickle
    with open(os.path.join(args.model_dir, "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
        processor.user_encoder = encoders["user"]
        processor.item_encoder = encoders["item"]

    # Create sequences
    user_sequences = processor.create_sequences(df_encoded)
    _, _, test_seqs = processor.split_sequences(user_sequences)

    # Create test dataset
    num_items = df_encoded["item_id"].max() + 1
    pos_items_by_user = {u: set(seq) for u, seq in user_sequences.items()}

    test_dataset = RecDataset(test_seqs, num_items, config.max_seq_len,
                              pos_items_by_user, mode="test", neg_samples=99)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Load model
    print("\n2. Loading model...")
    model = SASRec(
        num_items=num_items,
        hidden_dim=config.hidden_dim,
        max_seq_len=config.max_seq_len,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        dropout=config.dropout
    )

    model = load_model(model, os.path.join(args.model_dir, f"{args.model_name}.pth"), config.device)

    # Evaluate
    print("\n3. Evaluating...")
    evaluator = Evaluator(config)
    metrics = evaluator.evaluate(model, test_loader)

    print(f"\nResults:")
    print(f"  HR@{config.top_k}: {metrics['HR@K']:.4f}")
    print(f"  NDCG@{config.top_k}: {metrics['NDCG@K']:.4f}")
    print(f"  Precision@{config.top_k}: {metrics['Precision@K']:.4f}")
    print(f"  MRR@{config.top_k}: {metrics['MRR@K']:.4f}")

    # Split evaluation (cold vs warm)
    if args.eval_cold_warm:
        cold_threshold = 3
        cold_users = {u for u, seq in user_sequences.items() if len(seq) <= cold_threshold}
        warm_users = {u for u, seq in user_sequences.items() if len(seq) > cold_threshold}

        cold_test = {u: v for u, v in test_seqs.items() if u in cold_users}
        warm_test = {u: v for u, v in test_seqs.items() if u in warm_users}

        print(f"\nCold users ({len(cold_test)}):")
        if cold_test:
            cold_dataset = RecDataset(cold_test, num_items, config.max_seq_len,
                                      pos_items_by_user, mode="test", neg_samples=99)
            cold_loader = DataLoader(cold_dataset, batch_size=config.batch_size, shuffle=False)
            cold_metrics = evaluator.evaluate(model, cold_loader)
            print(f"  HR@{config.top_k}: {cold_metrics['HR@K']:.4f}")
            print(f"  NDCG@{config.top_k}: {cold_metrics['NDCG@K']:.4f}")

        print(f"\nWarm users ({len(warm_test)}):")
        if warm_test:
            warm_dataset = RecDataset(warm_test, num_items, config.max_seq_len,
                                      pos_items_by_user, mode="test", neg_samples=99)
            warm_loader = DataLoader(warm_dataset, batch_size=config.batch_size, shuffle=False)
            warm_metrics = evaluator.evaluate(model, warm_loader)
            print(f"  HR@{config.top_k}: {warm_metrics['HR@K']:.4f}")
            print(f"  NDCG@{config.top_k}: {warm_metrics['NDCG@K']:.4f}")


def rl_finetune_mode(args):
    """Fine-tune model with reinforcement learning."""
    print("=" * 80)
    print("RL FINE-TUNING MODE")
    print("=" * 80)

    # Load config
    config = load_config(os.path.join(args.model_dir, "config.json"))
    config.device = args.device
    config.rl_epochs = args.rl_epochs
    config.rl_lr = args.rl_lr
    config.entropy_coeff = args.entropy_coeff
    set_seed(config.seed)

    # Load data
    print("\n1. Loading data...")
    processor = DataProcessor(config)
    df = processor.load_data(args.data_path)
    df_filtered = processor.preprocess(df)
    df_encoded, _, _ = processor.encode_ids(df_filtered)

    # Load encoders
    import pickle
    with open(os.path.join(args.model_dir, "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
        processor.user_encoder = encoders["user"]
        processor.item_encoder = encoders["item"]

    # Create sequences
    user_sequences = processor.create_sequences(df_encoded)
    _, val_seqs, test_seqs = processor.split_sequences(user_sequences)

    # Create datasets
    num_items = df_encoded["item_id"].max() + 1
    pos_items_by_user = {u: set(seq) for u, seq in user_sequences.items()}

    val_dataset = RecDataset(val_seqs, num_items, config.max_seq_len,
                             pos_items_by_user, mode="val", neg_samples=99)
    test_dataset = RecDataset(test_seqs, num_items, config.max_seq_len,
                              pos_items_by_user, mode="test", neg_samples=99)

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Load model
    print("\n2. Loading model...")
    model = SASRec(
        num_items=num_items,
        hidden_dim=config.hidden_dim,
        max_seq_len=config.max_seq_len,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        dropout=config.dropout
    )

    model = load_model(model, os.path.join(args.model_dir, f"{args.model_name}.pth"), config.device)

    # Evaluate before RL
    print("\n3. Evaluation before RL...")
    evaluator = Evaluator(config)
    before_metrics = evaluator.evaluate(model, test_loader)
    print(f"  HR@{config.top_k}: {before_metrics['HR@K']:.4f}")
    print(f"  NDCG@{config.top_k}: {before_metrics['NDCG@K']:.4f}")

    # Run RL fine-tuning
    print(f"\n4. Running RL fine-tuning for {config.rl_epochs} epochs...")
    rl_trainer = RLTrainer(model, config)
    history = rl_trainer.finetune(val_loader, config.rl_epochs)

    # Evaluate after RL
    print("\n5. Evaluation after RL...")
    after_metrics = evaluator.evaluate(model, test_loader)
    print(f"  HR@{config.top_k}: {after_metrics['HR@K']:.4f} "
          f"(Δ: {after_metrics['HR@K'] - before_metrics['HR@K']:+.4f})")
    print(f"  NDCG@{config.top_k}: {after_metrics['NDCG@K']:.4f} "
          f"(Δ: {after_metrics['NDCG@K'] - before_metrics['NDCG@K']:+.4f})")

    # Save fine-tuned model
    torch.save(model.state_dict(),
               os.path.join(config.model_dir, f"{args.model_name}_rl.pth"))
    print(f"\nFine-tuned model saved as {args.model_name}_rl.pth")


def inference_mode(args):
    """Run inference and show recommendations."""
    print("=" * 80)
    print("INFERENCE MODE")
    print("=" * 80)

    # Load config
    config = load_config(os.path.join(args.model_dir, "config.json"))
    config.device = args.device
    set_seed(config.seed)

    # Load data
    print("\n1. Loading data...")
    processor = DataProcessor(config)
    df = processor.load_data(args.data_path)
    df_filtered = processor.preprocess(df)
    df_encoded, _, _ = processor.encode_ids(df_filtered)

    # Load encoders
    import pickle
    with open(os.path.join(args.model_dir, "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
        processor.user_encoder = encoders["user"]
        processor.item_encoder = encoders["item"]

    # Create sequences
    user_sequences = processor.create_sequences(df_encoded)

    # Load model
    print("\n2. Loading model...")
    num_items = df_encoded["item_id"].max() + 1

    model = SASRec(
        num_items=num_items,
        hidden_dim=config.hidden_dim,
        max_seq_len=config.max_seq_len,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        dropout=config.dropout
    )

    model_path = os.path.join(args.model_dir, f"{args.model_name}.pth")
    if args.use_rl and os.path.exists(os.path.join(args.model_dir, f"{args.model_name}_rl.pth")):
        model_path = os.path.join(args.model_dir, f"{args.model_name}_rl.pth")
        print("  Using RL fine-tuned model")

    model = load_model(model, model_path, config.device)

    # Load metadata if provided
    metadata = None
    if args.metadata_path:
        print("\n3. Loading metadata...")
        metadata_df = pd.read_csv(args.metadata_path)
        if "item" in metadata_df.columns and "title" in metadata_df.columns:
            metadata = metadata_df.set_index("item")["title"].to_dict()

    # Create inference object
    inference = RecommendationInference(model, processor, config)

    # Show recommendations
    print("\n4. Generating recommendations...")

    if args.user_id:
        # Specific user
        inference.display_recommendations(args.user_id, user_sequences,
                                          k=args.top_k, metadata=metadata)
    else:
        # Random users
        import random
        all_users = list(df_filtered["user"].unique())
        sample_users = random.sample(all_users, min(args.num_samples, len(all_users)))

        for user in sample_users:
            inference.display_recommendations(user, user_sequences,
                                              k=args.top_k, metadata=metadata)
            print("")


def main():
    parser = argparse.ArgumentParser(description="Modular Recommendation System")
    parser.add_argument("mode", choices=["train", "eval", "inference", "rl_finetune", "transfer"],
                        help="Mode to run the system in")

    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the CSV data file")
    parser.add_argument("--metadata_path", type=str,
                        help="Path to metadata CSV (for inference)")
    parser.add_argument("--column_mapping", type=str,
                        help='JSON string for column mapping, e.g., \'{"userId": "user", "movieId": "item"}\'')

    # Model arguments
    parser.add_argument("--model_dir", type=str, default="models",
                        help="Directory to save/load models")
    parser.add_argument("--model_name", type=str, default="best_model",
                        help="Name for the model file")
    parser.add_argument("--save_dir", type=str, default="outputs",
                        help="Directory to save outputs")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Hidden dimension size")
    parser.add_argument("--num_blocks", type=int, default=2,
                        help="Number of transformer blocks")
    parser.add_argument("--num_heads", type=int, default=2,
                        help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.4,
                        help="Dropout rate")

    # Transfer learning arguments
    parser.add_argument("--source_model_dir", type=str,
                        help="Directory of source model (for transfer)")
    parser.add_argument("--source_model_name", type=str, default="best_model",
                        help="Name of source model file")
    parser.add_argument("--bridge_hidden", type=int, default=128,
                        help="Hidden size for transfer bridge")

    # RL arguments
    parser.add_argument("--rl_epochs", type=int, default=50,
                        help="Number of RL fine-tuning epochs")
    parser.add_argument("--rl_lr", type=float, default=3e-5,
                        help="RL learning rate")
    parser.add_argument("--entropy_coeff", type=float, default=0.015,
                        help="Entropy coefficient for RL")

    # Evaluation arguments
    parser.add_argument("--eval_cold_warm", action="store_true",
                        help="Evaluate cold vs warm users separately")

    # Inference arguments
    parser.add_argument("--user_id", type=str,
                        help="Specific user ID for inference")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of sample users for inference")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of recommendations to show")
    parser.add_argument("--use_rl", action="store_true",
                        help="Use RL fine-tuned model if available")

    # General arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Run appropriate mode
    if args.mode == "train":
        train_mode(args)
    elif args.mode == "eval":
        eval_mode(args)
    elif args.mode == "inference":
        inference_mode(args)
    elif args.mode == "rl_finetune":
        rl_finetune_mode(args)
    elif args.mode == "transfer":
        transfer_mode(args)


if __name__ == "__main__":
    main()