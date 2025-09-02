import os
import argparse
import pickle
import json
import pandas as pd
import torch
import numpy as np
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from config import Config
from utils import (set_seed, save_config, load_model, load_config,
                   compute_user_representations, build_transfer_matrix)
from data_loader import DataProcessor
from dataset import SASRecDataset, TransferDataset
from models import SASRec, SASRecTransfer, init_target_from_source
from train import Trainer, TrainerTransfer
from evaluate import Evaluator
from inference import RecommendationInference
from rl_train import RLTrainer
from get_amazon_data import load_review_data, load_meta_data



def train_mode(args):
    """Train a model from scratch."""
    print("\n")
    print("=" * 80)
    print("TRAINING SASRec MODEL FROM SCRATCH")
    print("=" * 80)

    # Setup config
    config = Config(
        data_path=args.data_path,
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
    # Save config
    save_config(config, f"{config.save_dir}/source_domain/config.json")
    set_seed(config.seed)

    # Load and process data
    print("\nLoading and processing data...")
    processor = DataProcessor(config)
    df = processor.load_csv(
        filepath=args.data_path,
        max_items=args.max_items,
        seed=config.seed
    )

    df_filtered = processor.preprocess(df)
    df_encoded, user_enc, item_enc = processor.encode_ids(df_filtered)

    # Create sequences
    print("\nCreating user interaction sequences...")
    user_sequences = processor.create_sequences(df_encoded)
    train_seqs, val_seqs, test_seqs = processor.split_sequences(user_sequences)

    # Save encoders
    with open(os.path.join(f"{config.save_dir}/source_domain/encoders.pkl"), "wb") as f:
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
    results_df.to_csv(f"{config.save_dir}/source_domain/training_log.csv", index=False)

    # Final evaluation on test set
    print("\nEvaluating best model on test set at multiple K...")
    evaluator = Evaluator(config)
    model = load_model(model, f"{config.model_dir}/source_domain/{args.model_name}.pth")
    ks = [5, 10, 20, 50]
    original_k = getattr(config, "top_k", None)

    # Collect per-K metrics; K will be the DataFrame index
    rows_by_k = {}
    for k in ks:
        config.top_k = k
        mk = evaluator.evaluate(model, test_dataloader)
        rows_by_k[k] = {
            "HR": mk.get("HR@K"),
            "NDCG": mk.get("NDCG@K"),
            "Precision": mk.get("Precision@K"),
            "MRR": mk.get("MRR@K"),
        }

    # Restore original K
    if original_k is not None:
        config.top_k = original_k

    test_results_df = pd.DataFrame.from_dict(rows_by_k, orient="index").sort_index()
    test_results_df.index.name = "K"
    print("\nTest results table by K:")
    print(test_results_df.round(4))

    # Save final results
    test_results_df.to_csv(f"{config.save_dir}/source_domain/results_at_k.csv", index=False)

    print(f"\nAll artifacts saved to {config.save_dir}/source_domain/")


def transfer_mode(args):
    """Train with transfer learning from source domain."""
    print("\n")
    print("=" * 80)
    print("TRANSFER LEARNING MODE")
    print("=" * 80)

    # Load source config
    config = Config()
    source_config = load_config(f"{config.save_dir}/source_domain/config.json")

    # Setup target config
    config = Config(
        data_path=args.target_data_path,
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

    save_config(config, f"{config.save_dir}/transfer_domain/config.json")
    set_seed(config.seed)

    # Ensure source data path is provided
    if not args.source_data_path:
        raise ValueError("--source-data-path is required for transfer learning.")
    if not args.target_data_path:
        raise ValueError("--target-data-path is required for transfer learning.")
    if not os.path.exists("artifacts/source_domain/encoders.pkl"):
        raise FileNotFoundError("Source encoders not found, please ensure source model has been trained.")
    if not os.path.exists(f"models/source_domain/{args.source_model_name}.pth"):
        raise FileNotFoundError("Source model weights not found, please ensure source model has been trained.")

    # Load source model
    print("\nComputing source user representations...")
    with open("artifacts/source_domain/encoders.pkl", "rb") as f:
        source_encoders = pickle.load(f)

    source_processor = DataProcessor(source_config)
    source_df = source_processor.load_csv(
        filepath=args.source_data_path,
        max_items=source_config.max_items,
        seed=source_config.seed
    )
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
        batch_size=args.batch_size,
        save_path=os.path.join(config.save_dir, "source_domain", "user_representations.pkl")
    )

    # Load and process target data
    print("\nLoading and processing target model...")
    processor = DataProcessor(config)
    target_df = processor.load_csv(
        filepath=args.target_data_path,
        max_items=args.target_max_items,
        seed=config.seed
    )
    target_filtered_filtered = processor.preprocess(target_df)
    target_df_encoded, target_user_enc, target_item_enc = processor.encode_ids(target_filtered_filtered)

    with open(f"{config.save_dir}/transfer_domain/encoders.pkl", "wb") as f:
        pickle.dump({"user": target_user_enc, "item": target_item_enc}, f)

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
    torch.save(transfer_matrix, f"{config.save_dir}/source_domain/transfer_matrix.pth")

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
    print(f"   Target Val: {len(target_val_dataset)} samples")
    print(f"   Target Test: {len(target_test_dataset)} samples")

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

    # Save all training logs
    epochs_range = list(range(1, config.epochs + 1))
    results_df = pd.DataFrame({
        "epoch": epochs_range,
        "train_loss": transfer_train_losses,
        "val_loss": transfer_val_losses,
        "val_HR@K": [m["HR@K"] for m in transfer_val_metrics],
        "val_NDCG@K": [m["NDCG@K"] for m in transfer_val_metrics],
        "val_MRR@K": [m["MRR@K"] for m in transfer_val_metrics],
    })
    results_df.to_csv(f"{config.save_dir}/transfer_domain/training_log.csv", index=False)

    # Final evaluation on target test set
    print("\nEvaluating best model on test set at multiple K...")
    evaluator = Evaluator(config)
    model = load_model(transfer_model, f"{config.model_dir}/transfer_domain/{args.model_name}.pth")
    ks = [5, 10, 20, 50]
    original_k = getattr(config, "top_k", None)

    # Collect per-K metrics; K will be the DataFrame index
    rows_by_k = {}
    for k in ks:
        config.top_k = k
        mk = evaluator.evaluate(model, target_test_loader)
        rows_by_k[k] = {
            "HR": mk.get("HR@K"),
            "NDCG": mk.get("NDCG@K"),
            "Precision": mk.get("Precision@K"),
            "MRR": mk.get("MRR@K"),
        }

    # Restore original K
    if original_k is not None:
        config.top_k = original_k

    test_results_df = pd.DataFrame.from_dict(rows_by_k, orient="index").sort_index()
    test_results_df.index.name = "K"
    print("\nTest results table by K:")
    print(test_results_df.round(4))

    # Save final results
    test_results_df.to_csv(f"{config.save_dir}/transfer_domain/results_at_k.csv", index=False)
    print(f"\nAll artifacts saved to {config.save_dir}/transfer_domain/")



def eval_mode(args):
    """Evaluate a transfer model on all, cold-start, and warm-start users."""
    print("\n")
    print("=" * 80)
    print("EVALUATION MODE")
    print("=" * 80)

    # Load config
    config = Config()
    config = load_config(f"{config.save_dir}/transfer_domain/config.json")
    config.device = args.device
    set_seed(config.seed)

    # Load data
    print("\nLoading and processing data...")
    processor = DataProcessor(config)
    df = processor.load_csv(
        filepath=args.data_path,
        max_items=config.max_items,
        seed=config.seed
    )
    df_filtered = processor.preprocess(df)
    df_encoded, _, _ = processor.encode_ids(df_filtered)

    # Load encoders
    with open(os.path.join(args.save_dir, "transfer_domain", "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
        processor.user_encoder = encoders["user"]
        processor.item_encoder = encoders["item"]

    # Create sequences
    user_sequences = processor.create_sequences(df_encoded)
    train_seqs, _, test_seqs = processor.split_sequences(user_sequences)

    # Create test dataset
    num_items = df_encoded["item_id"].max() + 1
    num_users = df_encoded["user_id"].max() + 1
    pos_items_by_user = {u: set(seq) for u, seq in user_sequences.items()}
    transfer_matrix = torch.load(os.path.join(args.save_dir, "source_domain", "transfer_matrix.pth"))

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

    print(f"\nAll Users test Results:")
    print(f"   HR@{config.top_k}: {metrics['HR@K']:.4f}")
    print(f"   NDCG@{config.top_k}: {metrics['NDCG@K']:.4f}")
    print(f"   Prec@{config.top_k}: {metrics['Precision@K']:.4f}")
    print(f"   MRR@{config.top_k}: {metrics['MRR@K']:.4f}")

    # Cold vs Warm start evaluation
    print(f"Cold vs warm start evaluation...")
    train_lens = np.array([len(seq) for seq in train_seqs.values()])
    print(f"   Train sequence lengths - Min: {train_lens.min()}, Max: {train_lens.max()}")

    # Percentile-based threshold
    # cold_threshold = 3
    # cold_users = {u for u, seq in user_sequences.items() if len(seq) <= cold_threshold}
    # warm_users = {u for u, seq in user_sequences.items() if len(seq) > cold_threshold}

    # Percentile-based threshold
    p = 30
    cold_threshold = max(1, int(np.percentile(train_lens, p)))
    print(f"   Cold user threshold (<= {p}th percentile): {cold_threshold} interactions")

    cold_users = {u for u, seq in train_seqs.items() if len(seq) <= cold_threshold}
    warm_users = set(train_seqs.keys()) - cold_users

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

    # Build a compact DataFrame of metrics and print/save
    metrics_order = ["HR@K", "NDCG@K", "Precision@K", "MRR@K"]
    results_df = pd.DataFrame.from_dict(results, orient="index")
    results_df = results_df.reindex(columns=[c for c in metrics_order if c in results_df.columns])

    print(f"\nResults summary (overall / cold_start / warm_start) at K = {config.top_k}:")
    print(results_df.round(4))

    with open(os.path.join(args.save_dir, "transfer_domain", "detailed_test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to {os.path.join(args.save_dir, 'transfer_domain', 'detailed_test_results.json')}")



def rl_finetune_mode(args):
    """Fine-tune model with reinforcement learning."""
    print("=" * 80)
    print("REINFORCEMENT LEARNING FINE-TUNING MODE")
    print("=" * 80)

    # Load config
    config = load_config(os.path.join(args.save_dir, "transfer_domain", "config.json"))
    config.device = args.device
    config.rl_epochs = args.rl_epochs
    config.rl_lr = args.rl_lr
    config.entropy_coeff = args.entropy_coeff
    set_seed(args.seed)

    # Load data
    print(f"\nLoading and processing data...")
    processor = DataProcessor(config)
    df = processor.load_csv(
        filepath=args.data_path,
        max_items=config.max_items,
        seed=config.seed
    )
    df_filtered = processor.preprocess(df)
    df_encoded, _, _ = processor.encode_ids(df_filtered)

    # Load encoders
    with open(f"{config.save_dir}/transfer_domain_rl/encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
        processor.user_encoder = encoders["user"]
        processor.item_encoder = encoders["item"]

    # Create sequences
    user_sequences = processor.create_sequences(df_encoded)
    _, val_seqs, test_seqs = processor.split_sequences(user_sequences)

    # Create datasets
    num_items = df_encoded["item_id"].max() + 1
    pos_items_by_user = {u: set(seq) for u, seq in user_sequences.items()}

    val_dataset = SASRecDataset(val_seqs, num_items, pos_items_by_user=pos_items_by_user,
                                max_seq_len=config.max_seq_len, mode="val", neg_samples=config.neg_samples_eval)
    test_dataset = SASRecDataset(test_seqs, num_items, pos_items_by_user=pos_items_by_user,
                                 max_seq_len=config.max_seq_len, mode="test", neg_samples=config.neg_samples_eval)

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Load model
    print("\nLoading model and starting RL fine-tuning...")
    model = SASRec(
        num_items=num_items,
        hidden_dim=config.hidden_dim,
        max_seq_len=config.max_seq_len,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        dropout=config.dropout
    )

    model = load_model(model, os.path.join(args.model_dir, "transfer_domain", f"{args.model_name}.pth"), config.device)

    # Evaluate before RL fine-tuning
    evaluator = Evaluator(config)
    before_metrics = evaluator.evaluate(model, test_loader)
    print(f"   HR@{config.top_k}: {before_metrics['HR@K']:.4f}, NDCG@{config.top_k}: {pre_rl_metrics['NDCG@K']:.4f}")

    # RL Fine-tuning
    rl_trainer = RLTrainer(model, config)
    history = rl_trainer.finetune(val_loader, config.rl_epochs)

    # Evaluate after RL
    print("\n   Evaluating model after RL fine-tuning...")
    after_metrics = evaluator.evaluate(model, test_loader)
    print(f"   HR@{config.top_k}: {after_metrics['HR@K']:.4f} "
          f"(Δ: {after_metrics['HR@K'] - before_metrics['HR@K']:+.4f})")
    print(f"   NDCG@{config.top_k}: {after_metrics['NDCG@K']:.4f} "
          f"(Δ: {after_metrics['NDCG@K'] - before_metrics['NDCG@K']:+.4f})")


def inference_mode(args):
    """Run inference and show recommendations."""
    print("\n")
    print("=" * 80)
    print("INFERENCE MODE")
    print("=" * 80)

    # Load config
    config = Config()
    config = load_config(f"{config.save_dir}/transfer_domain/config.json")
    config.device = args.device
    set_seed(config.seed)

    transfer_matrix_path = f"{config.save_dir}/source_domain/transfer_matrix.pth"
    if not os.path.exists(transfer_matrix_path):
        raise FileNotFoundError(f"Transfer matrix not found at {transfer_matrix_path}. "
                                f"Please ensure transfer learning has been completed.")
    model_path = f"{config.model_dir}/transfer_domain/{args.model_name}.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}. "
                                f"Please ensure transfer learning has been completed.")

    # Load data
    print("\nLoading and processing data...")
    processor = DataProcessor(config)
    df = processor.load_csv(
        filepath=args.data_path,
        max_items=config.max_items,
        seed=config.seed
    )
    df_filtered = processor.preprocess(df)
    df_encoded, _, _ = processor.encode_ids(df_filtered)
    with open(f"{config.save_dir}/transfer_domain/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
        processor.user_encoder = encoders["user"]
        processor.item_encoder = encoders["item"]

    # Create sequences
    user_sequences = processor.create_sequences(df_encoded)
    num_items = df_encoded["item_id"].max() + 1
    num_users = df_encoded["user_id"].max() + 1
    pos_items_by_user = {u: set(seq) for u, seq in user_sequences.items()}

    # Load model
    print("\nLoading transfer model components...")
    transfer_matrix = torch.load(transfer_matrix_path, map_location=config.device)
    print(f"   Transfer matrix loaded with shape {transfer_matrix.shape}")

    # Check transfer matrix coverage
    coverage = (transfer_matrix.norm(dim=1) > 0).sum().item()
    print(f"   Transfer coverage: {coverage}/{num_users} users "
          f"({coverage / num_users * 100:.2f}%)")

    # Create base model
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

    # Inference
    inference = RecommendationInference(
        model=model,
        transfer_matrix=transfer_matrix,
        data_processor=processor,
        config=config
    )

    # Load weights
    model = load_model(model, model_path, config.device)
    inference.model = model

    metadata = None
    if args.metadata_path:
        print("\nLoading metadata...")
        metadata_df = pd.read_csv(args.metadata_path)

        # Check if specified columns exist
        if args.item_col not in metadata_df.columns:
            raise ValueError(f"Metadata item column '{args.item_col}' not found in {args.metadata_path}.")
        if args.title_col not in metadata_df.columns:
            raise ValueError(f"Metadata title column '{args.title_col}' not found in {args.metadata_path}.")

        # Convert the title column to string type and fill missing values (NaNs)
        metadata_df[args.title_col] = metadata_df[args.title_col].astype(str).fillna("No Title Available")
        metadata = metadata_df.set_index(args.item_col)[args.title_col].to_dict()
        print(f"   Loaded metadata for {len(metadata)} items.")

    print("\nGenerating recommendations...")
    if args.user_id:
        inference.display_recommendations(args.user_id, user_sequences, k=args.top_k, metadata=metadata)
    else:
        print("\n   Analyzing user groups for sampling...")
        all_user_ids = np.array(list(user_sequences.keys()))

        # Create a boolean mask for users with transfer info
        has_transfer_mask = np.zeros(num_users, dtype=bool)
        transfer_norms = transfer_matrix.norm(dim=1).cpu().numpy()
        active_transfer_indices = np.where(transfer_norms > 0)[0]
        has_transfer_mask[active_transfer_indices] = True

        # Apply the mask to our list of active users
        user_has_transfer = has_transfer_mask[all_user_ids]

        # Get the raw user names back
        ids_with_transfer = all_user_ids[user_has_transfer]
        ids_without_transfer = all_user_ids[~user_has_transfer]

        users_with_transfer = processor.user_encoder.inverse_transform(ids_with_transfer)
        users_without_transfer = processor.user_encoder.inverse_transform(ids_without_transfer)

        # Sampling logic
        n_samples = min(args.num_samples, len(all_user_ids))
        n_with = min(n_samples // 2, len(users_with_transfer))
        n_without = min(n_samples - n_with, len(users_without_transfer))

        sample_users = []
        if n_with > 0:
            sample_users.extend(np.random.choice(users_with_transfer, n_with, replace=False))
        if n_without > 0:
            sample_users.extend(np.random.choice(users_without_transfer, n_without, replace=False))

        print(f"   Sampling {len(sample_users)} users ({n_with} with transfer, {n_without} without transfer)")

        for user in sample_users:
            inference.display_recommendations(user, user_sequences, k=args.top_k, metadata=metadata)
            print("")

    # Summary statistics
    print("\n" + "=" * 80)
    print("TRANSFER MODEL STATISTICS")
    print("=" * 80)
    transfer_coverage = (transfer_matrix.norm(dim=1) > 0).sum().item()
    print(f"Total users: {num_users}")
    print(f"Users with source domain info: {transfer_coverage} ({transfer_coverage / num_users * 100:.1f}%)")
    print(f"Users without source domain info: {num_users - transfer_coverage} ({(num_users - transfer_coverage) / num_users * 100:.1f}%)")

    # Calculate average transfer vector magnitude for users that have it
    nonzero_mask = transfer_matrix.norm(dim=1) > 0
    if nonzero_mask.any():
        avg_magnitude = transfer_matrix[nonzero_mask].norm(dim=1).mean().item()
        print(f"Average transfer vector magnitude: {avg_magnitude:.4f}")


def get_amazon_data(args):
    """Download and preprocess Amazon review data."""
    print("\n")
    print("=" * 80)
    print("GET AMAZON DATA")
    print("=" * 80)

    if not args.download_dir:
        raise ValueError("--download-dir is required to specify where to save the data.")

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found in environment variables. "
                         "Please set HF_TOKEN in your .env file.")

    if args.review_or_meta == "review":
        for dom in args.domains:
            load_review_data(
                domain=dom,
                token=hf_token,
                save_dir=args.download_dir,
            )
    else:
        for dom in args.domains:
            load_meta_data(
                domain=dom,
                token=hf_token,
                save_dir=args.download_dir,
            )


def main():
    parser = argparse.ArgumentParser(description="Recommender System")
    parser.add_argument("mode",
                        choices=["train", "transfer", "eval", "rl_finetune", "inference", "get_amazon_data"],
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
                        help="Max number of rows to read from the data file")
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
    parser.add_argument("--rl-epochs", type=int, default=10,
                        help="Number of RL fine-tuning epochs")
    parser.add_argument("--rl-lr", type=float, default=1e-4,
                        help="Learning rate for RL fine-tuning")
    parser.add_argument("--entropy-coeff", type=float, default=0.01,
                        help="Entropy coefficient for RL fine-tuning")

    # Inference arguments
    parser.add_argument("--metadata-path", type=str, default=None,
                        help="Path to item metadata CSV for displaying item titles")
    parser.add_argument("--item-col", type=str, default="item",
                        help="Column name for item IDs in the metadata CSV")
    parser.add_argument("--title-col", type=str, default="title",
                        help="Column name for item titles in the metadata CSV")
    parser.add_argument("--user-id", type=str, default=None,
                        help="User ID to generate recommendations for (if not provided, samples multiple users)")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of users to sample for recommendations if --user-id is not provided")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top recommendations to display")

    # General arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the computations on (e.g., 'cuda' or 'cpu')")

    # Download Amazon data
    parser.add_argument("--review-or-meta", choices=["review", "meta"], default="reviews",
                        help="Whether to download review data or metadata from Amazon")
    parser.add_argument("--download-dir", type=str, default="data",
                        help="Dir to save the downloaded Amazon review data")
    parser.add_argument("--domains", nargs="+", default=["Books", "Movies_and_TV"],
                        help="List of Amazon review domains to download")

    args = parser.parse_args()

    # Run the appropriate mode
    if args.mode == "train":
        train_mode(args)
    elif args.mode == "transfer":
        transfer_mode(args)
    elif args.mode == "eval":
        eval_mode(args)
    elif args.mode == "inference":
        inference_mode(args)
    elif args.mode == "rl_finetune":
        rl_finetune_mode(args)
    elif args.mode == "get_amazon_data":
        get_amazon_data(args)

if __name__ == "__main__":
    main()