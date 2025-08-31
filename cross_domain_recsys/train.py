import os
import time
import torch
import torch.nn as nn
import gc
from tqdm import tqdm
from config import Config
from evaluate import Evaluator


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.evaluator = Evaluator(self.config)

    def train_epoch(self, model, train_loader, loss_fn, optimizer, device="cpu"):
        """Train one epoch."""
        model.train()
        loss_num, loss_den = 0.0, 0

        for batch in tqdm(train_loader, desc="   [TRAIN]"):
            input_seq = batch["input_seq"].to(device).long()
            pos_items = batch["target"].to(device).long()
            neg_items = batch["neg_items"].to(device).long()

            # Create candidates: first column is positive, rest are negatives
            candidates = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1).long()
            logits = model(input_seq, candidate_items=candidates)
            labels = torch.cat([
                torch.ones_like(logits[:, :1]),
                torch.zeros_like(logits[:, 1:])
            ], dim=1)

            # Compute loss
            loss = loss_fn(logits.reshape(-1), labels.reshape(-1))
            batch_elems = logits.numel()
            loss_num += loss.item() * batch_elems
            loss_den += batch_elems

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss_num / max(1, loss_den)

    def train(self, model, train_loader, val_loader, epochs, save_name="model"):
        """Full training loop."""
        device = self.config.device
        top_k = self.config.top_k
        model.to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        train_losses, val_losses, val_metrics_log = [], [], []
        best_ndcg = 0.0
        best_epoch = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            t0 = time.time()

            # Train
            train_loss = self.train_epoch(model, train_loader, loss_fn, optimizer, device)
            train_losses.append(train_loss)

            # Evaluate
            eval_metrics = self.evaluator.evaluate(model, val_loader, device)
            val_losses.append(eval_metrics["loss"])
            val_metrics_log.append({m: eval_metrics[m] for m in ["HR@K", "NDCG@K", "Precision@K", "MRR@K"]})

            print(f"   Train loss: {train_loss:.4f} | Val loss: {eval_metrics['loss']:.4f}")
            print(f"   HR@{top_k}: {eval_metrics['HR@K']:.4f} | NDCG@{top_k}: {eval_metrics['NDCG@K']:.4f} | "
                  f"Prec@{top_k} {eval_metrics['Precision@K']:.4f} | MRR@{top_k} {eval_metrics['MRR@K']:.4f}")
            print(f"   Time: {time.time() - t0:.2f}s")

            # Save best model
            if eval_metrics['NDCG@K'] > best_ndcg:
                best_ndcg = eval_metrics['NDCG@K']
                best_epoch = epoch + 1
                torch.save(
                    model.state_dict(),
                    os.path.join(self.config.model_dir, "source_domain", f"{save_name}.pth")
                )
                print(f"   -> New best model saved")

        print(f"\n   Training complete. Best epoch: {best_epoch} with NDCG@{self.config.top_k}: {best_ndcg:.4f}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return train_losses, val_losses, val_metrics_log, best_ndcg


class TrainerTransfer:
    def __init__(self, config: Config):
        self.config = config
        self.evaluator = Evaluator(self.config)

    def train_epoch_transfer(self, model, train_loader, loss_fn, optimizer, device="cpu"):
        """Train one epoch on transfer model."""
        model.train()
        loss_num, loss_den = 0.0, 0

        for batch in tqdm(train_loader, desc="   [TRAIN TRANSFER]"):
            input_seq = batch["input_seq"].to(device).long()
            pos_items = batch["target"].to(device).long()
            neg_items = batch["neg_items"].to(device).long()
            transfer = batch["transfer_src"].to(device)

            # Create candidates: first column is positive, rest are negatives
            logits = model(input_seq, transfer_source=transfer)
            pos_emb = model.base_model.item_embed(pos_items)
            neg_emb = model.base_model.item_embed(neg_items)

            # Compute scores
            pos_logits = (logits * pos_emb).sum(dim=1)
            neg_logits = torch.bmm(neg_emb, logits.unsqueeze(-1)).squeeze(-1)
            all_logits = torch.cat([
                pos_logits.unsqueeze(1),
                neg_logits],
            dim=1)
            all_labels = torch.cat([
                torch.ones_like(pos_logits).unsqueeze(1),
                torch.zeros_like(neg_logits)
            ], dim=1)

            # Compute loss
            loss = loss_fn(all_logits.reshape(-1), all_labels.reshape(-1))
            batch_elems = all_logits.numel()
            loss_num += loss.item() * batch_elems
            loss_den += batch_elems

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss_num / max(1, loss_den)

    def train_transfer(self, model, train_loader, val_loader, epochs, save_name="transfer_model"):
        """Full training loop for transfer model."""
        device = self.config.device
        top_k = self.config.top_k
        model.to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        train_losses, val_losses, val_metrics_log = [], [], []
        best_ndcg = 0.0
        best_epoch = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            t0 = time.time()

            # Train
            train_loss = self.train_epoch_transfer(model, train_loader, loss_fn, optimizer, device)
            train_losses.append(train_loss)

            # Evaluate
            eval_metrics = self.evaluator.evaluate_transfer(model, val_loader, device)
            val_losses.append(eval_metrics["loss"])
            val_metrics_log.append({m: eval_metrics[m] for m in ["HR@K", "NDCG@K", "Precision@K", "MRR@K"]})

            print(f"   Train loss: {train_loss:.4f} | Val loss: {eval_metrics['loss']:.4f}")
            print(f"   HR@{top_k}: {eval_metrics['HR@K']:.4f} | NDCG@{top_k}: {eval_metrics['NDCG@K']:.4f} | "
                  f"Prec@{top_k} {eval_metrics['Precision@K']:.4f} | MRR@{top_k} {eval_metrics['MRR@K']:.4f}")
            print(f"   Time: {time.time() - t0:.2f}s")

            # Save best model
            if eval_metrics['NDCG@K'] > best_ndcg:
                best_ndcg = eval_metrics['NDCG@K']
                best_epoch = epoch + 1
                torch.save(
                    model.state_dict(),
                    os.path.join(self.config.model_dir, "transfer_domain", f"{save_name}.pth")
                )
                print(f"   -> New best transfer model saved")

        print(f"\n   Training complete. Best epoch: {best_epoch} with NDCG@{self.config.top_k}: {best_ndcg:.4f}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return train_losses, val_losses, val_metrics_log, best_ndcg