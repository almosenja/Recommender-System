import os, argparse, json, torch, numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn

from cross_domain_recsys.config import PATHS, DATA, TRAIN, MODEL, PROJ
from cross_domain_recsys.utils import set_seed, get_device, ensure_dirs, count_params
from cross_domain_recsys.data import (load_amazon_reviews, preprocess_dataset, label_encoder,
                         create_user_sequences, sequences_loo_split)
from cross_domain_recsys.datasets import SASRecDataset, SASRecDatasetCD
from cross_domain_recsys.models.sasrec import SASRec
from cross_domain_recsys.models.transfer import SASRecCD
from cross_domain_recsys.train import (sasrec_trainer, train_sasrec_epoch, evaluate_sasrec,
                          train_target_with_transfer)
from cross_domain_recsys.recommend import sample_recommendations
from cross_domain_recsys.rl import PolicyGradientTrainer
from cross_domain_recsys.train import evaluate_transfer

def build_loaders(df_encoded, max_seq_len, num_items, batch_size, neg_train, neg_eval, cd=False, transfer_src_mat=None):
    user_sequences = create_user_sequences(df_encoded)
    pos_items_by_user = {u: set(seq) for u, seq in user_sequences.items()}
    train_sequences, val_sequences, test_sequences = sequences_loo_split(user_sequences)

    if cd:
        train_dataset = SASRecDatasetCD(train_sequences, num_items, transfer_src_mat, max_seq_len=max_seq_len, mode="train", neg_samples=neg_train)
        val_dataset = SASRecDatasetCD(val_sequences, num_items, transfer_src_mat, max_seq_len=max_seq_len, mode="val", neg_samples=neg_eval)
        test_dataset = SASRecDatasetCD(test_sequences, num_items, transfer_src_mat, max_seq_len=max_seq_len, mode="test", neg_samples=neg_eval)
    else:
        train_dataset = SASRecDataset(train_sequences, num_items, pos_items_by_user=pos_items_by_user, max_seq_len=max_seq_len, mode="train", neg_samples=neg_train)
        val_dataset = SASRecDataset(val_sequences, num_items, pos_items_by_user=pos_items_by_user, max_seq_len=max_seq_len, mode="val", neg_samples=neg_eval)
        test_dataset = SASRecDataset(test_sequences, num_items, pos_items_by_user=pos_items_by_user, max_seq_len=max_seq_len, mode="test", neg_samples=neg_eval)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return (train_loader, val_loader, test_loader,
            train_sequences, val_sequences, test_sequences, user_sequences)

def build_transfer_matrix(user_vecs_src, user_encoder_tgt, embed_dim, num_users_tgt):
    mat = np.zeros((num_users_tgt, embed_dim), dtype=np.float32)
    for raw_user, vec in user_vecs_src.items():
        if raw_user in user_encoder_tgt.classes_:
            uid_target = user_encoder_tgt.transform([raw_user])[0]
            mat[uid_target] = vec
    return torch.tensor(mat)

@torch.no_grad()
def compute_user_reprs_from_sequences(model_src, train_seqs_src, user_encoder_src, max_seq_len=50, device="cpu"):
    model_src.eval().to(device)
    user_vecs = {}
    for user_id, seq in train_seqs_src.items():
        if len(seq) < 1: continue
        seq = seq[-max_seq_len:]
        pad_len = max_seq_len - len(seq)
        input_seq = torch.tensor([([0] * pad_len + seq)], dtype=torch.long, device=device)
        hidden = model_src(input_seq)
        last_hidden = hidden[0, -1, :].squeeze(0)
        raw_user = user_encoder_src.inverse_transform([user_id])[0]
        user_vecs[raw_user] = last_hidden.detach().cpu().numpy()
    print(f"[xfer] built {len(user_vecs)} user vectors from source")
    return user_vecs

def cmd_train_source(args):
    set_seed(TRAIN.seed)
    device = get_device(TRAIN.device)
    ensure_dirs(PATHS.data_dir, PATHS.model_dir, PATHS.tb_dir)

    df = load_amazon_reviews(args.source, max_items=args.max_items)
    df = preprocess_dataset(df, DATA.min_user_interactions, DATA.min_item_interactions)
    df_enc, user_enc, item_enc, dom_enc = label_encoder(df, DATA.shift_item_id)

    num_users = int(df_enc["user_id"].max()) + 1
    num_items = int(df_enc["item_id"].max()) + 1
    print(f"[source] users={num_users} items={num_items}")

    loaders = build_loaders(df_enc, DATA.max_seq_len, num_items, TRAIN.batch_size, TRAIN.neg_samples_train, TRAIN.neg_samples_eval, cd=False)
    train_loader, val_loader, _, *_ = loaders

    model = SASRec(num_items=num_items, hidden_dim=MODEL.hidden_dim, max_seq_len=DATA.max_seq_len,
                   num_blocks=MODEL.num_blocks, num_heads=MODEL.num_heads, dropout=MODEL.dropout)
    print(f"[source] SASRec params: {count_params(model):,}")
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN.lr, weight_decay=TRAIN.weight_decay)

    best = sasrec_trainer(model, train_loader, val_loader, TRAIN.epochs, loss_fn, opt, k=TRAIN.k_eval, device=device, save_dir=PATHS.model_dir)

    # persist encoders & meta
    meta = {"user_classes": user_enc.classes_.tolist(), "item_classes": item_enc.classes_.tolist(),
            "num_items": num_items, "num_users": num_users, "domain": args.source}
    with open(os.path.join(PATHS.model_dir, "source_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    torch.save(model.state_dict(), os.path.join(PATHS.model_dir, "last_model_src.pth"))
    print("[source] done. best NDCG@{}={:.4f}".format(TRAIN.k_eval, best))

def cmd_train_target_transfer(args):
    set_seed(TRAIN.seed); device = get_device(TRAIN.device)
    # SOURCE
    df_s = load_amazon_reviews(args.source, max_items=args.max_items_src)
    df_s = preprocess_dataset(df_s, DATA.min_user_interactions, DATA.min_item_interactions)
    df_s_enc, user_enc_s, item_enc_s, dom_enc_s = label_encoder(df_s, DATA.shift_item_id)
    num_items_s = int(df_s_enc["item_id"].max()) + 1
    loaders_s = build_loaders(df_s_enc, DATA.max_seq_len, num_items_s, TRAIN.batch_size, TRAIN.neg_samples_train, TRAIN.neg_samples_eval, cd=False)
    train_loader_s, val_loader_s, test_loader_s, train_seqs_s, _, _, _ = loaders_s

    # train source model or load existing best
    model_src = SASRec(num_items=num_items_s, hidden_dim=MODEL.hidden_dim, max_seq_len=DATA.max_seq_len,
                       num_blocks=MODEL.num_blocks, num_heads=MODEL.num_heads, dropout=MODEL.dropout)
    src_ckpt = os.path.join(PATHS.model_dir, "best_model_src.pth")
    if os.path.exists(src_ckpt):
        model_src.load_state_dict(torch.load(src_ckpt, map_location="cpu"))
        print("[xfer] loaded existing source checkpoint")
    else:
        loss_fn = nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(model_src.parameters(), lr=TRAIN.lr, weight_decay=TRAIN.weight_decay)
        sasrec_trainer(model_src, train_loader_s, val_loader_s, TRAIN.epochs, loss_fn, opt, k=TRAIN.k_eval, device=device, save_dir=PATHS.model_dir)

    # TARGET
    df_t = load_amazon_reviews(args.target, max_items=args.max_items_tgt)
    df_t = preprocess_dataset(df_t, DATA.min_user_interactions, DATA.min_item_interactions)
    df_t_enc, user_enc_t, item_enc_t, dom_enc_t = label_encoder(df_t, DATA.shift_item_id)
    num_items_t = int(df_t_enc["item_id"].max()) + 1

    # transfer matrix
    user_vecs = compute_user_reprs_from_sequences(model_src, train_seqs_s, user_enc_s, max_seq_len=DATA.max_seq_len, device=device)
    transfer_src_mat = build_transfer_matrix(user_vecs, user_enc_t, MODEL.hidden_dim, int(df_t_enc["user_id"].max()) + 1)

    loaders_t = build_loaders(df_t_enc, DATA.max_seq_len, num_items_t, TRAIN.batch_size, TRAIN.neg_samples_train, TRAIN.neg_samples_eval, cd=True, transfer_src_mat=transfer_src_mat)
    train_loader_t, val_loader_t, test_loader_t, train_seqs_t, val_seqs_t, test_seqs_t, user_sequences_t = loaders_t

    # wrapper & train
    sasrec_target = SASRec(num_items=num_items_t, hidden_dim=MODEL.hidden_dim, max_seq_len=DATA.max_seq_len,
                           num_blocks=MODEL.num_blocks, num_heads=MODEL.num_heads, dropout=MODEL.dropout)
    transfer_model = SASRecCD(sasrec_target, hidden_dim=MODEL.hidden_dim, bridge_hidden=MODEL.bridge_hidden,
                              dropout=MODEL.dropout, fusion_mode=MODEL.fusion_mode).to(device)
    best = train_target_with_transfer(transfer_model, train_loader_t, val_loader_t, TRAIN.epochs, lr=TRAIN.lr, wd=TRAIN.weight_decay, k=TRAIN.k_eval, device=device)

    # save
    torch.save(transfer_model.state_dict(), os.path.join(PATHS.model_dir, "transfer_last.pth"))
    meta = {
        "user_classes_tgt": user_enc_t.classes_.tolist(),
        "item_classes_tgt": item_enc_t.classes_.tolist(),
        "user_classes_src": user_enc_s.classes_.tolist(),
        "num_items_tgt": num_items_t,
        "num_items_src": num_items_s,
        "domains": {"source": args.source, "target": args.target}
    }
    with open(os.path.join(PATHS.model_dir, "transfer_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    torch.save(transfer_src_mat, os.path.join(PATHS.model_dir, "transfer_src_mat.pt"))
    print("[transfer] done. best NDCG@{}={:.4f}".format(TRAIN.k_eval, best))

def cmd_train_target_baseline(args):
    set_seed(TRAIN.seed); device = get_device(TRAIN.device)
    df_t = load_amazon_reviews(args.target, max_items=args.max_items_tgt)
    df_t = preprocess_dataset(df_t, DATA.min_user_interactions, DATA.min_item_interactions)
    df_t_enc, user_enc_t, item_enc_t, dom_enc_t = label_encoder(df_t, DATA.shift_item_id)
    num_items_t = int(df_t_enc["item_id"].max()) + 1

    loaders_t = build_loaders(df_t_enc, DATA.max_seq_len, num_items_t, TRAIN.batch_size, TRAIN.neg_samples_train, TRAIN.neg_samples_eval, cd=False)
    train_loader_t, val_loader_t, test_loader_t, *_ = loaders_t

    model = SASRec(num_items=num_items_t, hidden_dim=MODEL.hidden_dim*2, max_seq_len=DATA.max_seq_len,
                   num_blocks=MODEL.num_blocks, num_heads=MODEL.num_heads, dropout=MODEL.dropout).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN.lr, weight_decay=TRAIN.weight_decay)

    best = 0.0; best_epoch = 0
    for epoch in range(TRAIN.epochs):
        tr = train_sasrec_epoch(model, train_loader_t, loss_fn, opt, device=device)
        ev = evaluate_sasrec(model, val_loader_t, loss_fn, k=TRAIN.k_eval, device=device)
        if ev["NDCG@K"] > best:
            best = ev["NDCG@K"]; best_epoch = epoch+1
            torch.save(model.state_dict(), os.path.join(PATHS.model_dir, "baseline_target_only_best.pth"))
        print(f"Epoch {epoch+1}/{TRAIN.epochs}  Train {tr:.4f}  Val {ev['Val loss']:.4f}  "
              f"HR@{TRAIN.k_eval} {ev['HR@K']:.4f}  NDCG@{TRAIN.k_eval} {ev['NDCG@K']:.4f}  "
              f"Prec@{TRAIN.k_eval} {ev['Precision@K']:.4f}  MRR@{TRAIN.k_eval} {ev['MRR@K']:.4f}  "
              f"{'(new best)' if ev['NDCG@K']==best and best_epoch==epoch+1 else ''}")
    print(f"[baseline] best epoch {best_epoch} NDCG@{TRAIN.k_eval}={best:.4f}")

def cmd_eval_compare(args):
    set_seed(TRAIN.seed); device = get_device(TRAIN.device)
    # load target data/encoders
    df_t = load_amazon_reviews(args.target, max_items=args.max_items_tgt)
    df_t = preprocess_dataset(df_t, DATA.min_user_interactions, DATA.min_item_interactions)
    df_t_enc, user_enc_t, item_enc_t, dom_enc_t = label_encoder(df_t, DATA.shift_item_id)
    num_items_t = int(df_t_enc["item_id"].max()) + 1

    # loaders for CD (needs transfer mat)
    transfer_src_mat = torch.load(os.path.join(PATHS.model_dir, "transfer_src_mat.pt"))
    loaders_t = build_loaders(df_t_enc, DATA.max_seq_len, num_items_t, TRAIN.batch_size, TRAIN.neg_samples_train, TRAIN.neg_samples_eval, cd=True, transfer_src_mat=transfer_src_mat)
    _, _, test_loader_t, train_seqs_t, *_ = loaders_t

    # baseline
    baseline = SASRec(num_items=num_items_t, hidden_dim=MODEL.hidden_dim*2, max_seq_len=DATA.max_seq_len,
                      num_blocks=MODEL.num_blocks, num_heads=MODEL.num_heads, dropout=MODEL.dropout).to(device)
    baseline.load_state_dict(torch.load(os.path.join(PATHS.model_dir, "baseline_target_only_best.pth"), map_location=device))
    baseline.eval()

    # transfer
    sasrec_target = SASRec(num_items=num_items_t, hidden_dim=MODEL.hidden_dim, max_seq_len=DATA.max_seq_len,
                           num_blocks=MODEL.num_blocks, num_heads=MODEL.num_heads, dropout=MODEL.dropout)
    transfer_model = SASRecCD(sasrec_target, hidden_dim=MODEL.hidden_dim, bridge_hidden=MODEL.bridge_hidden,
                              dropout=MODEL.dropout, fusion_mode=MODEL.fusion_mode).to(device)
    transfer_model.load_state_dict(torch.load(os.path.join(PATHS.model_dir, "transfer_best.pth"), map_location=device))
    transfer_model.eval()

    m_base = evaluate_sasrec(baseline, test_loader_t, nn.BCEWithLogitsLoss(), k=TRAIN.k_eval, device=device)
    m_xfer = evaluate_transfer(transfer_model, test_loader_t, nn.BCEWithLogitsLoss(), k=TRAIN.k_eval, device=device)
    print(json.dumps({"baseline": m_base, "transfer": m_xfer}, indent=2))

def cmd_recommend(args):
    set_seed(TRAIN.seed); device = get_device(TRAIN.device)
    # load meta
    meta = json.load(open(os.path.join(PATHS.model_dir, "transfer_meta.json"), "r", encoding="utf-8"))
    user_classes = np.array(meta["user_classes_tgt"])
    item_classes = np.array(meta["item_classes_tgt"])
    # Encoders reconstructed on the fly for inverse_transform
    class DummyEnc:
        def __init__(self, classes): self.classes_ = classes
        def transform(self, xs):
            idxs = []
            for x in xs:
                idxs.append(int(np.where(self.classes_ == x)[0][0]))
            return np.array(idxs, dtype=np.int64)
        def inverse_transform(self, xs):
            return np.array([self.classes_[i] for i in xs])
    user_enc_t = DummyEnc(user_classes)
    item_enc_t = DummyEnc(item_classes)

    # load target dataset to build sequences
    df_t = load_amazon_reviews(args.target, max_items=args.max_items_tgt)
    df_t = preprocess_dataset(df_t, DATA.min_user_interactions, DATA.min_item_interactions)
    df_t_enc, _, _, _ = label_encoder(df_t, DATA.shift_item_id)
    num_items_t = int(df_t_enc["item_id"].max()) + 1
    user_sequences = create_user_sequences(df_t_enc)

    # load models
    use_transfer = args.use_transfer
    if use_transfer:
        transfer_src_mat = torch.load(os.path.join(PATHS.model_dir, "transfer_src_mat.pt"))
        sasrec_target = SASRec(num_items=num_items_t, hidden_dim=MODEL.hidden_dim, max_seq_len=DATA.max_seq_len,
                               num_blocks=MODEL.num_blocks, num_heads=MODEL.num_heads, dropout=MODEL.dropout)
        model = SASRecCD(sasrec_target, hidden_dim=MODEL.hidden_dim, bridge_hidden=MODEL.bridge_hidden,
                         dropout=MODEL.dropout, fusion_mode=MODEL.fusion_mode).to(device)
        model.load_state_dict(torch.load(os.path.join(PATHS.model_dir, "transfer_best.pth"), map_location=device))
        xfer_mat = transfer_src_mat
    else:
        model = SASRec(num_items=num_items_t, hidden_dim=MODEL.hidden_dim*2, max_seq_len=DATA.max_seq_len,
                       num_blocks=MODEL.num_blocks, num_heads=MODEL.num_heads, dropout=MODEL.dropout).to(device)
        model.load_state_dict(torch.load(os.path.join(PATHS.model_dir, "baseline_target_only_best.pth"), map_location=device))
        xfer_mat = None

    recs = sample_recommendations(args.user, k=args.k, model=model, user_encoder=user_enc_t,
                                  item_encoder=item_enc_t, sequences=user_sequences,
                                  xfer_mat=xfer_mat, max_len=DATA.max_seq_len, device=device)
    print(json.dumps({"user": args.user, "k": args.k, "recs": recs}, indent=2))

def cmd_rl_demo(args):
    set_seed(TRAIN.seed); device = get_device(TRAIN.device)
    # load target data/encoders
    df_t = load_amazon_reviews(args.target, max_items=args.max_items_tgt)
    df_t = preprocess_dataset(df_t, DATA.min_user_interactions, DATA.min_item_interactions)
    df_t_enc, user_enc_t, item_enc_t, dom_enc_t = label_encoder(df_t, DATA.shift_item_id)
    num_items_t = int(df_t_enc["item_id"].max()) + 1

    # loaders (CD) and model
    transfer_src_mat = torch.load(os.path.join(PATHS.model_dir, "transfer_src_mat.pt"))
    train_loader_t, val_loader_t, test_loader_t, *_ = build_loaders(df_t_enc, DATA.max_seq_len, num_items_t, TRAIN.batch_size, TRAIN.neg_samples_train, TRAIN.neg_samples_eval, cd=True, transfer_src_mat=transfer_src_mat)

    sasrec_target = SASRec(num_items=num_items_t, hidden_dim=MODEL.hidden_dim, max_seq_len=DATA.max_seq_len,
                           num_blocks=MODEL.num_blocks, num_heads=MODEL.num_heads, dropout=MODEL.dropout)
    model = SASRecCD(sasrec_target, hidden_dim=MODEL.hidden_dim, bridge_hidden=MODEL.bridge_hidden,
                     dropout=MODEL.dropout, fusion_mode=MODEL.fusion_mode).to(device)
    model.load_state_dict(torch.load(os.path.join(PATHS.model_dir, "transfer_best.pth"), map_location=device))

    print("[RL] before:", evaluate_transfer(model, test_loader_t, nn.BCEWithLogitsLoss(), k=TRAIN.k_eval, device=device))
    rl = PolicyGradientTrainer(model, lr=5e-5, entropy_coeff=0.01, temperature=1.0, baseline_momentum=0.9, device=device)
    rl.offline_demo_finetune(val_loader_t, steps=args.steps, sample_actions=True)
    print("[RL]  after:", evaluate_transfer(model, test_loader_t, nn.BCEWithLogitsLoss(), k=TRAIN.k_eval, device=device))

def main():
    ap = argparse.ArgumentParser(description="Cross-domain SASRec (modular)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_src = sub.add_parser("train-source", help="Train SASRec on source domain")
    ap_src.add_argument("--source", default=PROJ.source_domain)
    ap_src.add_argument("--max_items", type=int, default=None)

    ap_xfer = sub.add_parser("train-target-transfer", help="Train target with transfer from source")
    ap_xfer.add_argument("--source", default=PROJ.source_domain)
    ap_xfer.add_argument("--target", default=PROJ.target_domain)
    ap_xfer.add_argument("--max_items_src", type=int, default=None)
    ap_xfer.add_argument("--max_items_tgt", type=int, default=None)

    ap_base = sub.add_parser("train-target-baseline", help="Train target-only baseline SASRec")
    ap_base.add_argument("--target", default=PROJ.target_domain)
    ap_base.add_argument("--max_items_tgt", type=int, default=None)

    ap_eval = sub.add_parser("eval-compare", help="Evaluate baseline vs transfer on target TEST")
    ap_eval.add_argument("--target", default=PROJ.target_domain)
    ap_eval.add_argument("--max_items_tgt", type=int, default=None)

    ap_rec = sub.add_parser("recommend", help="Sample recommendations for a given raw user id on target")
    ap_rec.add_argument("--target", default=PROJ.target_domain)
    ap_rec.add_argument("--user", required=True)
    ap_rec.add_argument("--k", type=int, default=10)
    ap_rec.add_argument("--use-transfer", action="store_true")
    ap_rec.add_argument("--max_items_tgt", type=int, default=None)

    ap_rl = sub.add_parser("rl-demo", help="Offline RL demo fine-tune on VAL")
    ap_rl.add_argument("--target", default=PROJ.target_domain)
    ap_rl.add_argument("--steps", type=int, default=5)
    ap_rl.add_argument("--max_items_tgt", type=int, default=None)

    args = ap.parse_args()
    if args.cmd == "train-source": cmd_train_source(args)
    elif args.cmd == "train-target-transfer": cmd_train_target_transfer(args)
    elif args.cmd == "train-target-baseline": cmd_train_target_baseline(args)
    elif args.cmd == "eval-compare": cmd_eval_compare(args)
    elif args.cmd == "recommend": cmd_recommend(args)
    elif args.cmd == "rl-demo": cmd_rl_demo(args)

if __name__ == "__main__":
    main()