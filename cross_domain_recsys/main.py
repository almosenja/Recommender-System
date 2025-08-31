import os
import argparse
import pickle
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from config import Config
from utils import (set_seed, save_config, load_model, load_config, compute_user_representations,
                   build_transfer_matrix, load_user_representations)
from data_loader import DataProcessor
from dataset import SASRecDataset, TransferDataset
from models import SASRec, SASRecTransfer, init_target_from_source
from train import Trainer, TrainerTransfer
from evaluate import Evaluator



def train_mode(args):
    """Train a model from scratch."""
    print("\n")
    print("=" * 80)
    print("TRAINING SASRec MODEL FROM SCRATCH")
    print("=" * 80)

    # Setup config
    config = Config(
        data_path=args.data_path,
        save_dir=args.save_dir,
        model_dir=args.model_dir,
        max_items=args.max_items,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
    )

    set_seed(config.seed)

    # Load and process data
    print("\nLoading and processing data...")
    processor = DataProcessor(config)
    df = processor.load_csv(filepath=args.data_path, max_items=args.max_items)

    df_filtered = processor.preprocess(df)
    df_encoded, user_enc, item_enc = processor.encode_ids(df_filtered)

    # Create sequences
    print("\nCreating user interaction sequences...")
    user_sequences = processor.create_sequences(df_encoded)
    train_seqs, val_seqs, test_seqs = processor.split_sequences(user_sequences)

    # Save encoders
    with open(os.path.join(config.save_dir, "source_domain", "encoders.pkl"), "wb") as f:
        pickle.dump({"user": user_enc, "item": item_enc}, f)

    # Create datasets and dataloaders
    print("\nCreating datasets and dataloaders...")
    num_items = df_encoded["item_id"].max() + 1
    pos_items_by_user = {u: set(seq) for u, seq in user_sequences.items()}

    train_dataset = SASRecDataset(train_seqs, num_items, pos_items_by_user=pos_items_by_user,
                                  max_seq_len=config.max_seq_len, mode="train", neg_samples=config.neg_samples_train)
    val_dataset = SASRecDataset(val_seqs, num_items, pos_items_by_user=pos_items_by_user,
                                max_seq_len=config.max_seq_len, mode="val", neg_samples=config.neg_samples_eval)
    test_dataset = SASRecDataset(test_seqs, num_items, pos_items_by_user=pos_items_by_user,
                                 max_seq_len=config.max_seq_len, mode="test", neg_samples=config.neg_samples_eval)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")
    print(f"   Test: {len(test_dataset)} samples")

    # Create and train model
    print("\nTraining model...")
    model = SASRec(
        num_items=num_items,
        hidden_dim=config.hidden_dim,
        max_seq_len=config.max_seq_len,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        dropout=config.dropout
    )

    trainer = Trainer(config)
    train_losses, val_losses, val_metrics, best_ndcg = trainer.train(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        epochs=config.epochs,
        save_name=args.model_name
    )

    # Final evaluation on test set
    print("\nEvaluating best model on test set...")
    evaluator = Evaluator(config)
    model = load_model(model, os.path.join(config.model_dir, "source_domain", f"{args.model_name}.pth"))
    test_metrics = evaluator.evaluate(model, test_dataloader)

    print(f"\nTest Results:")
    print(f"   HR@{config.top_k}: {test_metrics['HR@K']:.4f}")
    print(f"   NDCG@{config.top_k}: {test_metrics['NDCG@K']:.4f}")
    print(f"   MRR@{config.top_k}: {test_metrics['MRR@K']:.4f}")

    # Save training and validation results to csv
    epochs_range = list(range(1, config.epochs + 1))
    results_df = pd.DataFrame({
        "epoch": epochs_range,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_HR@K": [m["HR@K"] for m in val_metrics],
        "val_NDCG@K": [m["NDCG@K"] for m in val_metrics],
        "val_MRR@K": [m["MRR@K"] for m in val_metrics],
    })
    results_df.to_csv(os.path.join(config.save_dir, "source_domain", "training_log.csv"), index=False)

    # Save config
    save_config(config, os.path.join(config.save_dir, "source_domain", "config.json"))

    # Save final results
    with open(os.path.join(config.save_dir, "source_domain", "test_results.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nAll artifacts saved to {config.save_dir}/source_domain/")


def transfer_mode(args):
    """Train with transfer learning from source domain."""
    print("\n")
    print("=" * 80)
    print("TRANSFER LEARNING MODE")
    print("=" * 80)

    # If no source config provided, raise error
    if not args.save_dir or not os.path.exists(args.source_model_dir):
        raise ValueError("Source config and model directory must be provided for transfer learning.")

    source_config = load_config(os.path.join(args.save_dir, "source_domain", "config.json"))

    # Setup target config
    config = Config(
        data_path=args.target_data_path,
        save_dir=args.save_dir,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        max_items=args.target_max_items,
        max_seq_len=args.max_seq_len,
        hidden_dim=source_config.hidden_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        bridge_hidden=args.bridge_hidden,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device
    )

    set_seed(config.seed)

    # Load source model
    print("\nComputing source user representations (it may take a while)...")

    with open(os.path.join(args.save_dir, "source_domain", "encoders.pkl"), "rb") as f:
        source_encoders = pickle.load(f)

    # Ensure source data path is provided
    if not args.source_data_path:
        raise ValueError("--source-data-path is required to compute user representations.")

    source_processor = DataProcessor(source_config)
    source_df = source_processor.load_csv(filepath=args.source_data_path, max_items=config.max_items)
    source_df_filtered = source_processor.preprocess(source_df)
    source_df_encoded, _, _ = source_processor.encode_ids(source_df_filtered)

    # Get source user sequences
    source_sequences = source_processor.create_sequences(source_df_encoded)
    source_train_seqs, _, _ = source_processor.split_sequences(source_sequences)
    num_items_source = source_df_encoded["item_id"].max() + 1

    # Create source model to load weights
    source_model = SASRec(
        num_items=num_items_source,
        hidden_dim=source_config.hidden_dim,
        max_seq_len=source_config.max_seq_len,
        num_blocks=source_config.num_blocks,
        num_heads=source_config.num_heads,
        dropout=source_config.dropout
    )

    source_model = load_model(
        source_model,
        os.path.join(args.source_model_dir, "source_domain", f"{args.source_model_name}.pth"),
        device=source_config.device
    )

    # Compute user representations in source domain
    user_vecs_source = compute_user_representations(
        model=source_model,
        sequences=source_train_seqs,
        user_encoder_source=source_encoders["user"],
        max_seq_len=source_config.max_seq_len,
        device=source_config.device,
        batch_size=args.batch_size
    )

    # Load and process target data
    print("\nLoading and processing target model...")
    processor = DataProcessor(config)
    target_df = processor.load_csv(filepath=args.target_data_path, max_items=args.target_max_items)
    target_filtered_filtered = processor.preprocess(target_df)
    target_df_encoded, target_user_enc, target_item_enc = processor.encode_ids(target_filtered_filtered)

    # Analyze domain overlap
    users_src = set(source_df["user"])
    users_tgt = set(target_df["user"])
    common_users = users_src.intersection(users_tgt)
    print(f"   Source domain users: {len(users_src)}")
    print(f"   Target domain users: {len(users_tgt)}")
    print(f"   Common users: {len(common_users)}")
    print(f"   Common users (% of target): {len(common_users) / len(users_tgt) * 100:.2f}%")
    print(f"   Common users (% of source): {len(common_users) / len(users_src) * 100:.2f}%")

    # Create sequences
    target_sequences = processor.create_sequences(target_df_encoded)
    target_train_seqs, target_val_seqs, target_test_seqs = processor.split_sequences(target_sequences)

    # Build transfer matrix
    print("\nBuilding transfer matrix...")
    target_num_users = target_df_encoded["user_id"].max() + 1
    target_num_items = target_df_encoded["item_id"].max() + 1

    transfer_matrix = build_transfer_matrix(
        source_vecs=user_vecs_source,
        target_encoder=target_user_enc,
        num_users_target=target_num_users
    )

    # Save transfer matrix
    torch.save(transfer_matrix, os.path.join(config.save_dir, "source_domain", "transfer_matrix.pth"))

    # Create datasets with transfer
    print("\nCreating datasets with transfer information...")
    pos_items_by_user = {u: set(seq) for u, seq in target_sequences.items()}

    target_train_dataset = TransferDataset(
        target_train_seqs,
        target_num_items,
        config.max_seq_len,
        pos_items_by_user=pos_items_by_user,
        transfer_matrix=transfer_matrix,
        mode="train",
        neg_samples=config.neg_samples_train
    )

    target_val_dataset = TransferDataset(
        target_val_seqs,
        target_num_items,
        config.max_seq_len,
        pos_items_by_user=pos_items_by_user,
        transfer_matrix=transfer_matrix,
        mode="val",
        neg_samples=config.neg_samples_eval
    )

    target_test_dataset = TransferDataset(
        target_test_seqs,
        target_num_items,
        config.max_seq_len,
        pos_items_by_user=pos_items_by_user,
        transfer_matrix=transfer_matrix,
        mode="test",
        neg_samples=config.neg_samples_eval
    )

    target_train_loader = DataLoader(target_train_dataset, batch_size=config.batch_size, shuffle=True)
    target_val_loader = DataLoader(target_val_dataset, batch_size=config.batch_size, shuffle=False)
    target_test_loader = DataLoader(target_test_dataset, batch_size=config.batch_size, shuffle=False)

    print(f"   Target Train: {len(target_train_dataset)} samples")
    print(f"   Target Val:   {len(target_val_dataset)} samples")
    print(f"   Target Test:  {len(target_test_dataset)} samples")

    # Create and train transfer model
    print("\nTraining transfer model...")
    target_base = SASRec(
        num_items=target_num_items,
        hidden_dim=source_config.hidden_dim,
        max_seq_len=config.max_seq_len,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        dropout=config.dropout
    )

    # Initialize target model from source
    init_target_from_source(source_model, target_base)

    # Wrap with transfer capability
    transfer_model = SASRecTransfer(
        target_base=target_base,
        hidden_dim=source_config.hidden_dim,
        bridge_hidden=config.bridge_hidden,
        dropout=config.dropout
    )

    # Train the transfer model
    trainer = TrainerTransfer(config)
    transfer_train_losses, transfer_val_losses, transfer_val_metrics, transfer_best_ndcg = trainer.train_transfer(
        model=transfer_model,
        train_loader=target_train_loader,
        val_loader=target_val_loader,
        epochs=config.epochs,
        save_name=args.model_name
    )

    # Final evaluation on target test set
    print("\nEvaluating best transfer model on target test set...")
    best_transfer_model = load_model(transfer_model, os.path.join(config.model_dir, "transfer_domain", f"{args.model_name}.pth"), config.device)
    evaluator = Evaluator(config)
    test_metrics = evaluator.evaluate_transfer(best_transfer_model, target_test_loader)

    print(f"\nTarget Test Results:")
    print(f"   HR@{config.top_k}: {test_metrics['HR@K']:.4f}")
    print(f"   NDCG@{config.top_k}: {test_metrics['NDCG@K']:.4f}")
    print(f"   Prec@{config.top_k}: {test_metrics['Precision@K']:.4f}")
    print(f"   MRR@{config.top_k}: {test_metrics['MRR@K']:.4f}")

    # Save all artifacts
    epochs_range = list(range(1, config.epochs + 1))
    results_df = pd.DataFrame({
        "epoch": epochs_range,
        "train_loss": transfer_train_losses,
        "val_loss": transfer_val_losses,
        "val_HR@K": [m["HR@K"] for m in transfer_val_metrics],
        "val_NDCG@K": [m["NDCG@K"] for m in transfer_val_metrics],
        "val_MRR@K": [m["MRR@K"] for m in transfer_val_metrics],
    })
    results_df.to_csv(os.path.join(config.save_dir, "transfer_domain", "training_log.csv"), index=False)

    save_config(config, os.path.join(config.save_dir, "transfer_domain", "config.json"))
    with open(os.path.join(config.save_dir, "transfer_domain", "encoders.pkl"), "wb") as f:
        pickle.dump({"user": target_user_enc, "item": target_item_enc}, f)
    with open(os.path.join(config.save_dir, "transfer_domain", "test_results.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\nAll artifacts saved to {config.save_dir}/transfer_domain/")



def eval_mode(args):
    """Evaluate a transfer model on all, cold-start, and warm-start users."""
    print("=" * 80)
    print("EVALUATION MODE")
    print("=" * 80)

    # Load config
    config = load_config(os.path.join(args.model_dir, "transfer_domain", "config.json"))
    config.device = args.device
    set_seed(config.seed)

    # Load data
    print("\nLoading and processing data...")
    processor = DataProcessor(config)
    df = processor.load_csv(filepath=args.data_path, max_items=config.max_items)
    df_filtered = processor.preprocess(df)
    df_encoded, _, _ = processor.encode_ids(df_filtered)

    # Load encoders
    with open(os.path.join(args.save_dir, "artifacts", "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
        processor.user_encoder = encoders["user"]
        processor.item_encoder = encoders["item"]

    # Create sequences
    user_sequences = processor.create_sequences(df_encoded)
    _, _, test_seqs = processor.split_sequences(user_sequences)

    # Create test dataset
    num_items = df_encoded["item_id"].max() + 1
    num_users = df_encoded["user_id"].max() + 1
    pos_items_by_user = {u: set(seq) for u, seq in user_sequences.items()}
    transfer_matrix = torch.load(os.path.join(args.save_dir, "artifacts", "transfer_matrix.pth"))

    test_dataset = TransferDataset(
        test_seqs,
        num_items,
        config.max_seq_len,
        pos_items_by_user=pos_items_by_user,
        transfer_matrix=transfer_matrix,  # No transfer info for cold-start eval
        mode="test",
        neg_samples=config.neg_samples_eval
    )

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Create base and transfer model
    print("\nLoading model and evaluating...")
    base_model = SASRec(
        num_items=num_items,
        hidden_dim=config.hidden_dim,
        max_seq_len=config.max_seq_len,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        dropout=config.dropout
    )

    model = SASRecTransfer(
        base_model,
        hidden_dim=config.hidden_dim,
        bridge_hidden=config.bridge_hidden,
        dropout=config.dropout
    )

    # Load weights
    model = load_model(model, os.path.join(args.model_dir, "transfer_domain", f"{args.model_name}.pth"), config.device)

    # Evaluate
    evaluator = Evaluator(config)
    metrics = evaluator.evaluate_transfer(model, test_loader)

    print(f"\nTest Results:")
    print(f"   HR@{config.top_k}: {metrics['HR@K']:.4f}")
    print(f"   NDCG@{config.top_k}: {metrics['NDCG@K']:.4f}")
    print(f"   Prec@{config.top_k}: {metrics['Precision@K']:.4f}")
    print(f"   MRR@{config.top_k}: {metrics['MRR@K']:.4f}")

    # Cold vs Warm start evaluation
    print(f"Cold vs warm start evaluation...")
    cold_threshold = 3
    cold_users = {u for u, seq in user_sequences.items() if len(seq) <= cold_threshold}
    warm_users = {u for u, seq in user_sequences.items() if len(seq) > cold_threshold}
    cold_test = {u: seq for u, seq in test_seqs.items() if u in cold_users}
    warm_test = {u: seq for u, seq in test_seqs.items() if u in warm_users}
    print(f"   Test cold users: {len(cold_test)}, test warm users: {len(warm_test)}")

    transfer_coverage = (transfer_matrix.norm(dim=1) > 0).sum().item()
    print(f"   Transfer coverage: {transfer_coverage}/{num_users} users "
          f"({transfer_coverage / num_users * 100:.2f}%)")

    # Check cold user coverage
    cold_with_transfer=sum(1 for u in cold_users if u < len(transfer_matrix) and transfer_matrix[u].norm() > 0)
    print(f"   Cold users with transfer: {cold_with_transfer}/{len(cold_users)} "
          f"({cold_with_transfer / len(cold_users) * 100:.2f}%)")

    # Create cold and warm datasets
    cold_dataset = TransferDataset(
        cold_test,
        num_items,
        config.max_seq_len,
        pos_items_by_user=pos_items_by_user,
        transfer_matrix=transfer_matrix,
        mode="test",
        neg_samples=config.neg_samples_eval
    )

    warm_dataset = TransferDataset(
        warm_test,
        num_items,
        config.max_seq_len,
        pos_items_by_user=pos_items_by_user,
        transfer_matrix=transfer_matrix,
        mode="test",
        neg_samples=config.neg_samples_eval
    )

    test_loader_cold = DataLoader(cold_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader_warm = DataLoader(warm_dataset, batch_size=config.batch_size, shuffle=False)

    # Evaluate cold and warm
    cold_test = evaluator.evaluate_transfer(model, test_loader_cold)
    warm_test = evaluator.evaluate_transfer(model, test_loader_warm)

    # Save results
    results = {
        "overall": metrics,
        "cold_start": cold_test,
        "warm_start": warm_test
    }

    with open(os.path.join(args.save_dir, "transfer_domain", "detailed_test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to {os.path.join(args.save_dir, 'transfer_domain', 'detailed_test_results.json')}")




def rl_finetune_mode():
    pass

def inference_mode():
    pass

def main():
    parser = argparse.ArgumentParser(description="Recommender System")
    parser.add_argument("mode", choices=["train", "transfer", "eval", "rl_finetune", "inference"],
                        help="Mode to run the system in")

    # Data and paths
    parser.add_argument("--data-path", type=str,
                        help="Path to the CSV data file")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Directory to save/load model checkpoints")
    parser.add_argument("--save-dir", type=str, default="artifacts",
                        help="Directory to save artifacts like logs and configs")
    parser.add_argument("--model-name", type=str, default="best_model",
                        help="Name for saving the model checkpoint")

    # Target
    parser.add_argument("--target-data-path", type=str, default=None,
                        help="Path to the target domain CSV data file (for transfer learning)")
    parser.add_argument("--target-max-items", type=int, default=None,
                        help="Max items in target domain (for transfer)")

    # Transfer
    parser.add_argument("--source-data-path", type=str,
                        help="Path to the source domain CSV data file (for transfer learning)")
    parser.add_argument("--source-model-dir", type=str, default="models",
                        help="Directory of the source model (for transfer learning)")
    parser.add_argument("--source-model-name", type=str, default="best_model",
                        help="Name of the source model checkpoint for transfer learning")

    # Data filtering
    parser.add_argument("--max-items", type=int, default=None,
                        help="Max number of unique items to consider")
    parser.add_argument("--max-seq-len", type=int, default=50,
                        help="Max length of user interaction sequences")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size for training and evaluation")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-6,
                        help="Weight decay for optimizer")
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="Hidden dimension size for the model")
    parser.add_argument("--num-blocks", type=int, default=2,
                        help="Number of transformer blocks in the model")
    parser.add_argument("--num-heads", type=int, default=2,
                        help="Number of attention heads in the model")
    parser.add_argument("--dropout", type=float, default=0.4,
                        help="Dropout rate for the model")

    # Transfer model arguments
    parser.add_argument("--bridge-hidden", type=int, default=128,
                        help="Hidden dimension for the bridge layer in transfer model")

    # RL fine-tuning arguments

    # General arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the computations on (e.g., 'cuda' or 'cpu')")


    args = parser.parse_args()

    # Run the appropriate mode
    if args.mode == "train":
        train_mode(args)
    elif args.mode == "transfer":
        transfer_mode(args)

if __name__ == "__main__":
    main()