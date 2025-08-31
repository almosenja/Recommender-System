import torch
import torch.nn as nn


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(self.relu(self.w1(x))))


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = PointWiseFeedForward(hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.ln1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_out))
        return x


class SASRec(nn.Module):
    def __init__(self, num_items, hidden_dim=64, max_seq_len=50,
                 num_blocks=2, num_heads=2, dropout=0.2):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.item_embed = nn.Embedding(num_items, hidden_dim, padding_idx=0)
        self.positional_embed = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            AttentionBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])

        self.ln = nn.LayerNorm(hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.item_embed.weight[1:])
        nn.init.xavier_normal_(self.positional_embed.weight)

    def forward(self, input_seq, candidate_items=None):
        batch_size, seq_len = input_seq.shape

        item_embeds = self.item_embed(input_seq)
        positions = torch.arange(seq_len, device=input_seq.device).unsqueeze(0)
        pos_embeds = self.positional_embed(positions)
        x = self.dropout(item_embeds + pos_embeds)

        attn_mask = self._create_causal_mask(seq_len, input_seq.device)
        pad_mask = input_seq.eq(0)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln(x)
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        if candidate_items is not None:
            cand_emb = self.item_embed(candidate_items)
            last_hidden = x[:, -1, :].unsqueeze(1)
            scores = torch.matmul(last_hidden, cand_emb.transpose(1, 2)).squeeze(1)
            return scores

        return x

    def _create_causal_mask(self, seq_len, device):
        mask = torch.full((seq_len, seq_len), 0.0, device=device)
        mask = mask.masked_fill(
            torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool(),
            float('-inf')
        )
        return mask

    def predict_next(self, input_seq):
        seq_repr = self.forward(input_seq)
        last_hidden = seq_repr[:, -1, :]
        all_item_embeds = self.item_embed.weight
        scores = torch.matmul(last_hidden, all_item_embeds.T)
        return scores


class SASRecTransfer(nn.Module):
    """SASRec with cross-domain transfer capability."""

    def __init__(self, base_sasrec, hidden_dim, bridge_hidden, dropout):
        super().__init__()
        self.base = base_sasrec
        self.bridge = nn.Sequential(
            nn.Linear(hidden_dim, bridge_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bridge_hidden, hidden_dim)
        )
        self.linear = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, input_seq, transfer_src=None):
        seq_output = self.base(input_seq)
        last_hidden = seq_output[:, -1, :]

        if transfer_src is not None:
            bridge_out = self.bridge(transfer_src)
            has_transfer = (transfer_src.abs().sum(dim=-1, keepdim=True) > 0).float()

            last_hidden_n = nn.functional.layer_norm(last_hidden, last_hidden.shape[-1:])
            bridge_out_n = nn.functional.layer_norm(bridge_out, bridge_out.shape[-1:])

            combined = torch.cat([last_hidden_n, bridge_out_n], dim=-1)
            gate = torch.sigmoid(self.linear(combined))
            fused_logic = gate * last_hidden + (1.0 - gate) * bridge_out

            fused = has_transfer * fused_logic + (1.0 - has_transfer) * last_hidden
        else:
            fused = last_hidden

        return fused

    def predict_next(self, input_seq, transfer_src=None):
        fused_repr = self.forward(input_seq, transfer_src)
        all_item_embeds = self.base.item_embed.weight
        scores = torch.matmul(fused_repr, all_item_embeds.T)
        return scores