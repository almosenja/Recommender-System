import torch
import torch.nn as nn
from tqdm import tqdm


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.k = config.top_k

    @torch.no_grad()
    def evaluate(self, model, eval_loader, device=None):
        """Evaluate model on given data loader."""
        if device is None:
            device = self.config.device

        model.eval()
        model.to(device)
        loss_fn = nn.BCEWithLogitsLoss()

        total = 0
        sum_hr, sum_ndcg, sum_prec, sum_mrr = 0.0, 0.0, 0.0, 0.0
        loss_num, loss_den = 0.0, 0

        for batch in tqdm(eval_loader, desc="   [EVAL]"):
            input_seq = batch["input_seq"].to(device)
            pos_items = batch["target"].to(device)
            neg_items = batch["neg_items"].to(device)
            batch_size = input_seq.size(0)

            candidates = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1).long()
            logits = model(input_seq, candidate_items=candidates)
            labels = torch.cat([
                torch.ones_like(logits[:, :1]),
                torch.zeros_like(logits[:, 1:])
            ], dim=1)

            # Calculate metrics
            full_idx = torch.argsort(logits, dim=1, descending=True)
            rank = (full_idx == 0).nonzero(as_tuple=True)[1] + 1

            hit = (rank <= self.k).float()
            ndcg = torch.where(rank <= self.k, 1.0 / torch.log2(rank.float() + 1), torch.zeros_like(hit))
            precision = hit / float(self.k)
            mrr = torch.where(rank <= self.k, 1.0 / rank.float(), torch.zeros_like(hit))

            # Loss
            batch_elems = logits.numel()
            loss = loss_fn(logits.reshape(-1), labels.reshape(-1))
            loss_num += loss.item() * batch_elems
            loss_den += batch_elems

            # Accumulate
            sum_hr += hit.sum().item()
            sum_ndcg += ndcg.sum().item()
            sum_prec += precision.sum().item()
            sum_mrr += mrr.sum().item()
            total += batch_size

        return {
            "HR@K": sum_hr / total if total else 0.0,
            "NDCG@K": sum_ndcg / total if total else 0.0,
            "Precision@K": sum_prec / total if total else 0.0,
            "MRR@K": sum_mrr / total if total else 0.0,
            "loss": loss_num / loss_den if loss_den else 0.0
        }

    def evaluate_transfer(self, model, eval_loader, device=None):
        """Evaluate cross-domain model on given data loader."""
        if device is None:
            device = self.config.device

        model.eval()
        model.to(device)
        loss_fn = nn.BCEWithLogitsLoss()

        total = 0
        sum_hr, sum_ndcg, sum_prec, sum_mrr = 0.0, 0.0, 0.0, 0.0
        loss_num, loss_den = 0.0, 0

        for batch in tqdm(eval_loader, desc="   [EVAL TRANSFER]"):
            input_seq = batch["input_seq"].to(device)
            pos_items = batch["target"].to(device)
            neg_items = batch["neg_items"].to(device)
            transfer = batch["transfer_src"].to(device)
            batch_size = input_seq.size(0)

            fused = model(input_seq, transfer_source=transfer)
            cand = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)
            cand_emb = model.base_model.item_embed(cand)
            logits = torch.bmm(cand_emb, fused.unsqueeze(-1)).squeeze(-1)

            # Calculate metrics
            full_idx = torch.argsort(logits, dim=1, descending=True)
            rank = (full_idx == 0).nonzero(as_tuple=True)[1] + 1  # 1-based
            hit = (rank <= self.k).float()
            ndcg = torch.where(rank <= self.k, 1.0 / torch.log2(rank.float() + 1), torch.zeros_like(hit))
            precision = hit / float(self.k)
            mrr = 1.0 / rank.float()

            # Loss
            labels = torch.cat([
                torch.ones_like(logits[:, :1]),
                torch.zeros_like(logits[:, 1:])
            ], dim=1)

            batch_elems = logits.numel()
            loss = loss_fn(logits.reshape(-1), labels.reshape(-1))
            loss_num += loss.item() * batch_elems
            loss_den += batch_elems

            # Accumulate
            sum_hr += hit.sum().item()
            sum_ndcg += ndcg.sum().item()
            sum_prec += precision.sum().item()
            sum_mrr += mrr.sum().item()
            total += batch_size

        return {
            "HR@K": sum_hr / total if total else 0.0,
            "NDCG@K": sum_ndcg / total if total else 0.0,
            "Precision@K": sum_prec / total if total else 0.0,
            "MRR@K": sum_mrr / total if total else 0.0,
            "loss": loss_num / loss_den if loss_den else 0.0
        }