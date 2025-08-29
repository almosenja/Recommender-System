import os, time, torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from tqdm import tqdm

from .config import PATHS
from .utils import ensure_dirs

# ----------- core SASRec training / evaluation -----------

def train_sasrec_epoch(model, train_loader, loss_fn, optimizer, device="cpu"):
    model.train()
    total_loss = 0.0; n_batches = 0
    for batch in tqdm(train_loader, desc="Training"):
        input_seq = batch["input_seq"].to(device)
        pos_items = batch["target"].to(device)
        neg_items = batch["neg_items"].to(device)

        seq_output = model(input_seq)                  # [B, L, D]
        last_hidden = seq_output[:, -1, :]            # [B, D]
        pos_embeds = model.item_embed(pos_items)      # [B, D]
        neg_embeds = model.item_embed(neg_items)      # [B, N, D]

        pos_logits = (last_hidden * pos_embeds).sum(dim=1)               # [B]
        neg_logits = torch.bmm(neg_embeds, last_hidden.unsqueeze(-1)).squeeze(-1)  # [B, N]

        all_logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)       # [B, 1+N]
        all_labels = torch.cat([torch.ones_like(pos_logits).unsqueeze(1),
                                torch.zeros_like(neg_logits)], dim=1)               # [B, 1+N]

        loss = loss_fn(all_logits.reshape(-1), all_labels.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)

@torch.no_grad()
def evaluate_sasrec(model, eval_loader, loss_fn, k=10, device="cpu"):
    model.eval()
    total = 0
    sum_hr = 0.0
    sum_ndcg = 0.0
    sum_prec = 0.0
    sum_mrr = 0.0
    sum_val_loss = 0.0
    n_loss_batches = 0

    for batch in tqdm(eval_loader, desc="Evaluating"):
        input_seq = batch["input_seq"].to(device)
        target = batch["target"].to(device)
        neg_items = batch["neg_items"].to(device)

        seq_output = model(input_seq)
        last_hidden = seq_output[:, -1, :]
        candidates = torch.cat([target.unsqueeze(1), neg_items], dim=1)
        cand_emb = model.item_embed(candidates)
        scores = torch.bmm(cand_emb, last_hidden.unsqueeze(-1)).squeeze(-1)

        pos_scores = scores[:, 0]
        neg_scores = scores[:, 1:]
        labels = torch.cat([torch.ones_like(pos_scores).unsqueeze(1),
                            torch.zeros_like(neg_scores)], dim=1)
        batch_loss = loss_fn(scores.reshape(-1), labels.reshape(-1))
        sum_val_loss += batch_loss.item(); n_loss_batches += 1

        _, full_idx = torch.sort(scores, dim=1, descending=True)
        rank  = (full_idx == 0).nonzero(as_tuple=True)[1] + 1
        hit = (rank <= k).float()
        ndcg = torch.where(rank <= k, 1.0 / torch.log2(rank.float() + 1), torch.zeros_like(hit))
        precision = hit / float(k)
        mrr = torch.where(rank <= k, 1.0 / rank.float(), torch.zeros_like(hit))

        B = input_seq.size(0)
        sum_hr += hit.sum().item(); sum_ndcg += ndcg.sum().item()
        sum_prec += precision.sum().item(); sum_mrr += mrr.sum().item()
        total += B

    return {
        "HR@K": sum_hr / max(1,total),
        "NDCG@K": sum_ndcg / max(1,total),
        "Precision@K": sum_prec / max(1,total),
        "MRR@K": sum_mrr / max(1,total),
        "Val loss": sum_val_loss / max(1,n_loss_batches)
    }

def sasrec_trainer(model, train_loader, eval_loader, epochs, loss_fn, optimizer, k=10, device="cpu", save_dir=None):
    save_dir = save_dir or PATHS.model_dir
    ensure_dirs(save_dir, PATHS.tb_dir)
    model.to(device)
    writer = SummaryWriter(log_dir=PATHS.tb_dir)

    train_losses, val_losses = [], []
    best_ndcg, best_epoch = 0.0, 0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}:")
        t0 = time.time()
        train_loss = train_sasrec_epoch(model, train_loader, loss_fn, optimizer, device=device)
        train_losses.append(train_loss)
        m = evaluate_sasrec(model, eval_loader, loss_fn, k=k, device=device)
        val_losses.append(m["Val loss"])

        if m["NDCG@K"] > best_ndcg:
            best_ndcg = m["NDCG@K"]; best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model_src.pth"))
        torch.save(model.state_dict(), os.path.join(save_dir, "last_model_src.pth"))

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", m["Val loss"], epoch)
        writer.add_scalar(f"Metrics/Val_HR@{k}", m["HR@K"], epoch)
        writer.add_scalar(f"Metrics/Val_NDCG@{k}", m["NDCG@K"], epoch)
        writer.add_scalar(f"Metrics/Val_Precision@{k}", m["Precision@K"], epoch)
        writer.add_scalar(f"Metrics/Val_MRR@{k}", m["MRR@K"], epoch)

        print(f"Train {train_loss:.4f}  Val {m['Val loss']:.4f}  "
              f"HR@{k} {m['HR@K']:.4f}  NDCG@{k} {m['NDCG@K']:.4f}  "
              f"Prec@{k} {m['Precision@K']:.4f}  MRR@{k} {m['MRR@K']:.4f}  "
              f"{'(new best)' if m['NDCG@K']==best_ndcg and best_epoch==epoch+1 else ''}" 
              f"Time {time.time()-t0:.2f}s\n")

    print(f"[train] best epoch {best_epoch} NDCG@{k}={best_ndcg:.4f}")
    writer.close()
    return best_ndcg

# ----------- transfer training / evaluation -----------

def train_epoch_transfer(model, loader, loss_fn, optimizer, device="cpu"):
    model.train()
    total = 0.0
    n = 0
    for batch in tqdm(loader, desc="Training"):
        inp = batch["input_seq"].to(device)
        pos = batch["target"].to(device)
        neg = batch["neg_items"].to(device)
        transfer = batch["transfer_src"].to(device)

        fused = model(inp, transfer_src=transfer)
        pos_emb = model.base.item_embed(pos)
        neg_emb = model.base.item_embed(neg)

        pos_logits = (fused * pos_emb).sum(dim=1)
        neg_logits = torch.bmm(neg_emb, fused.unsqueeze(-1)).squeeze(-1)

        all_logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], 1)
        all_labels = torch.cat([torch.ones_like(pos_logits).unsqueeze(1),
                                torch.zeros_like(neg_logits)], 1)

        loss = loss_fn(all_logits.reshape(-1), all_labels.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(1,n)

@torch.no_grad()
def evaluate_transfer(model, loader, loss_fn, k=10, device="cpu"):
    model.eval()
    total = 0.0
    hits = 0.0
    ndcgs = 0.0
    precs = 0.0
    mrrs = 0.0
    loss_sum = 0.0
    nb = 0
    for batch in tqdm(loader, desc="Evaluating"):
        inp = batch["input_seq"].to(device)
        tgt = batch["target"].to(device)
        neg = batch["neg_items"].to(device)
        transfer = batch["transfer_src"].to(device)

        fused = model(inp, transfer_src=transfer)
        cand = torch.cat([tgt.unsqueeze(1), neg], dim=1)
        cand_emb = model.base.item_embed(cand)
        scores = torch.bmm(cand_emb, fused.unsqueeze(-1)).squeeze(-1)

        labels = torch.cat([torch.ones_like(scores[:, :1]), torch.zeros_like(scores[:, 1:])], dim=1)
        batch_loss = loss_fn(scores.reshape(-1), labels.reshape(-1))
        loss_sum += batch_loss.item(); nb += 1

        _, idx = torch.sort(scores, dim=1, descending=True)
        rank = (idx == 0).nonzero(as_tuple=True)[1] + 1
        hit = (rank <= k).float()
        ndcg = torch.where(rank <= k, 1.0 / torch.log2(rank.float() + 1), torch.zeros_like(hit))
        precision = hit / float(k)
        mrr = torch.where(rank <= k, 1.0 / rank.float(), torch.zeros_like(hit))

        B = inp.size(0)
        hits += hit.sum().item(); ndcgs += ndcg.sum().item()
        precs += precision.sum().item(); mrrs += mrr.sum().item()
        total += B

    return {
        "HR@K": hits / max(1,total),
        "NDCG@K": ndcgs / max(1,total),
        "Precision@K": precs / max(1,total),
        "MRR": mrrs / max(1,total),
        "Val loss": loss_sum / max(1,nb)
    }

def train_target_with_transfer(model, train_loader, val_loader, epochs, lr=1e-3, wd=1e-6, k=10, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()
    best_ndcg, best_epoch = 0.0, 0
    os.makedirs(PATHS.model_dir, exist_ok=True)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}:")
        tr = train_epoch_transfer(model, train_loader, loss_fn, opt, device=device)
        ev = evaluate_transfer(model, val_loader, loss_fn, k=k, device=device)
        if ev["NDCG@K"] > best_ndcg:
            best_ndcg, best_epoch = ev["NDCG@K"], epoch+1
            torch.save(model.state_dict(), os.path.join(PATHS.model_dir, "transfer_best.pth"))
        print(f"Train {tr:.4f}  Val {ev['Val loss']:.4f}  "
              f"HR@{k} {ev['HR@K']:.4f}  NDCG@{k} {ev['NDCG@K']:.4f}  "
              f"Prec@{k} {ev['Precision@K']:.4f}  MRR {ev['MRR']:.4f}  "
              f"{'(new best)' if ev['NDCG@K']==best_ndcg and best_epoch==epoch+1 else ''}")
    print(f"[transfer] best epoch {best_epoch} NDCG@{k}={best_ndcg:.4f}\n")
    return best_ndcg
