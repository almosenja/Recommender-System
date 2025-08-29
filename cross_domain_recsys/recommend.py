import torch

@torch.no_grad()
def sample_recommendations(
    raw_user,
    k,
    model,
    user_encoder,
    item_encoder,
    sequences,
    xfer_mat=None,       # None for baseline SASRec
    max_len=50,
    device="cpu",
    pop_items=None
):
    if raw_user not in user_encoder.classes_:
        return (pop_items or [])[:k]
    uid = int(user_encoder.transform([raw_user])[0])
    seq = sequences.get(uid, [])
    seen = set(seq)
    seq = seq[-max_len:]
    inp = torch.tensor([[0]*(max_len-len(seq)) + seq], dtype=torch.long, device=device)

    if hasattr(model, "base"):  # SASRecCD
        xfer = None if xfer_mat is None else xfer_mat[uid:uid+1].to(device)
        fused = model(inp, transfer_src=xfer)                 # [1, D]
        W = model.base.item_embed.weight                      # [I, D]
    else:
        seq_repr = model(inp)                                 # [1, L, D]
        fused = seq_repr[:, -1, :]
        W = model.item_embed.weight

    scores = fused @ W.T
    scores[:, 0] = -1e9
    if seen:
        scores[0, list(seen)] = -1e9

    topk_ids = torch.topk(scores, k, dim=1).indices.squeeze(0).tolist()
    rec_items = [item_encoder.inverse_transform([i-1])[0] for i in topk_ids]
    return rec_items