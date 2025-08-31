import torch
import torch.nn as nn
from tqdm import tqdm
import time
import os
from config import Config
from evaluators import Evaluator


class Trainer:
    def __init__(self, config: Config):
        self.config = config

    def train_epoch(self, model, train_loader, loss_fn, optimizer, device="cpu"):
        """Train one epoch."""
        model.train()
        loss_num, loss_den = 0.0, 0

        for batch in tqdm(train_loader, desc="  [TRAIN]"):
            input_seq = batch["input_seq"].to(device).long()
            pos_items = batch["target"].to(device).long()
            neg_items = batch["neg_items"].to(device).long()

            candidates = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1).long()
            logits = model(input_seq, candidate_items=candidates)
            labels = torch.cat([
                torch.ones_like(logits[:, :1]),
                torch.zeros_like(logits[:, 1:])
            ], dim=1)

            loss = loss_fn(logits.reshape(-1), labels.reshape(-1))
            batch_elems = logits.numel()
            loss_num += loss.item() * batch_elems
            loss_den += batch_elems

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        return loss_num / max(1, loss_den)

    def train(self, model, train_loader, val_loader, epochs, save_name="model"):
        """Full training loop."""
        device = self.config.device
        model.to(device)

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        evaluator = Evaluator(self.config)

        best_ndcg = 0.0
        best_epoch = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            t0 = time.time()

            # Train
            train_loss = self.train_epoch(model, train_loader, loss_fn, optimizer, device)

            # Evaluate
            metrics = evaluator.evaluate(model, val_loader)

            print(f"  Train loss: {train_loss:.4f} | Val loss: {metrics['loss']:.4f}")
            print(f"  HR@{self.config.top_k}: {metrics['HR@K']:.4f} | "
                  f"NDCG@{self.config.top_k}: {metrics['NDCG@K']:.4f}")
            print(f"  Time: {time.time() - t0:.2f}s")

            # Save best model
            if metrics['NDCG@K'] > best_ndcg:
                best_ndcg = metrics['NDCG@K']
                best_epoch = epoch + 1
                torch.save(
                    model.state_dict(),
                    os.path.join(self.config.model_dir, f"{save_name}.pth")
                )
                print(f"  ✓ New best model saved")

        print(f"\nTraining complete. Best epoch: {best_epoch} with NDCG@{self.config.top_k}: {best_ndcg:.4f}")
        return best_ndcg


class TransferTrainer(Trainer):
    """Trainer for cross-domain transfer learning."""

    def train_epoch_transfer(self, model, loader, loss_fn, optimizer, device="cpu"):
        """Train one epoch with transfer learning."""
        model.train()
        loss_num, loss_den = 0.0, 0

        for batch in tqdm(loader, desc="  [TRAIN Transfer]"):
            input_seq = batch["input_seq"].to(device).long()
            pos_items = batch["target"].to(device).long()
            neg_items = batch["neg_items"].to(device).long()
            transfer = batch.get("transfer_src", None)
            if transfer is not None:
                transfer = transfer.to(device)

            # Get fused representation
            logits = model(input_seq, transfer_src=transfer)
            pos_emb = model.base.item_embed(pos_items)
            neg_emb = model.base.item_embed(neg_items)

            # Compute logits
            pos_logits = (logits * pos_emb).sum(dim=1)
            neg_logits = torch.bmm(neg_emb, logits.unsqueeze(-1)).squeeze(-1)
            all_logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], 1)
            all_labels = torch.cat([
                torch.ones_like(pos_logits).unsqueeze(1),
                torch.zeros_like(neg_logits)
            ], 1)

            loss = loss_fn(all_logits.reshape(-1), all_labels.reshape(-1))
            batch_elems = logits.numel()
            loss_num += loss.item() * batch_elems
            loss_den += batch_elems

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss_num / max(1, loss_den)