import torch
import torch.nn as nn


# Building SASRec model
class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(self.relu(self.w1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.2):
        super().__init__()

        # Multi-head attention
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # Layer norms
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Feed-forward network
        self.ffn = PointWiseFeedForward(hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # Self-attention with residual connection
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.ln1(x + self.dropout(attn_out))

        # Feed-forward network with residual connection
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_out))

        return x

class SASRec(nn.Module):
    """ Self-Attentive Sequential Recommendation (SASRec) model."""
    def __init__(self,
                 num_items,
                 hidden_dim=64,
                 max_seq_len=50,
                 num_blocks=2,
                 num_heads=2,
                 dropout=0.2):
        super().__init__()

        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Embedding layers
        self.item_embed = nn.Embedding(num_items, hidden_dim, padding_idx=0)
        self.positional_embed = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Stack of SASRec blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout) for _ in range(num_blocks)
        ])

        # Final layer norm
        self.ln = nn.LayerNorm(hidden_dim)

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.item_embed.weight[1:])  # Skip padding idx
        nn.init.xavier_normal_(self.positional_embed.weight)

    def forward(self, input_seq, candidate_items=None):
        batch_size, seq_len = input_seq.shape

        # Get item embeddings
        item_embeds = self.item_embed(input_seq)  # [B, L, D]

        # Add positional embeddings
        positions = torch.arange(seq_len, device=input_seq.device).unsqueeze(0)
        pos_embeds = self.positional_embed(positions)  # [1, L, D]
        x = self.dropout(item_embeds + pos_embeds)

        # Create causal attention mask
        attn_mask = self._create_causal_mask(seq_len, input_seq.device)
        pad_mask = input_seq.eq(0)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        # Final layer norm
        x = self.ln(x)  # [B, L, D]
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        # If candidate_items provided, score them
        if candidate_items is not None:
            # Get embeddings for candidate items
            cand_emb = self.item_embed(candidate_items) # [B, N, D]

            # Use last position's representation for scoring
            last_hidden = x[:, -1, :].unsqueeze(1)  # [B, 1, D]

            # Compute scores via dot product
            scores = torch.matmul(last_hidden, cand_emb.transpose(1, 2)).squeeze(1) # [B, N]
            return scores

        return x

    def _create_causal_mask(self, seq_len, device):
        mask = torch.full((seq_len, seq_len), 0.0, device=device)
        mask = mask.masked_fill(torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool(), float('-inf'))
        return mask

    def predict_next(self, input_seq):
        # Get sequence representations
        seq_repr = self.forward(input_seq)  # [B, L, D]

        # Use last position for prediction
        last_hidden = seq_repr[:, -1, :]  # [B, D]

        # Score against all item embeddings
        all_item_embeds = self.item_embed.weight  # [num_items, D]
        scores = torch.matmul(last_hidden, all_item_embeds.T)  # [B, num_items]
        return scores

class SASRecTransfer(nn.Module):
    def __init__(self, target_base, hidden_dim, bridge_hidden, dropout):
        super().__init__()
        self.base_model = target_base
        self.bridge = nn.Sequential(
            nn.Linear(hidden_dim, bridge_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bridge_hidden, hidden_dim)
        )
        self.linear = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, input_seq, transfer_source=None):
        seq_output = self.base_model(input_seq)
        last_hidden = seq_output[:, -1, :]

        if transfer_source is not None:
            bridge_out = self.bridge(transfer_source)

            # A mask to identify which users in the batch have a source vector
            has_transfer = (transfer_source.abs().sum(dim=-1, keepdim=True) > 0).float()
            last_hidden_n = nn.functional.layer_norm(last_hidden, last_hidden.shape[-1:])
            bridge_out_n = nn.functional.layer_norm(bridge_out, bridge_out.shape[-1:])

            combined = torch.cat([last_hidden_n, bridge_out_n], dim=-1)
            gate = torch.sigmoid(self.linear(combined))
            fused_logic = gate * last_hidden + (1.0 - gate) * bridge_out
            fused = has_transfer * fused_logic + (1.0 - has_transfer) * last_hidden
        else:
            fused = last_hidden

        return fused

    def predict_next(self, input_seq, transfer_source=None):
        fused_repr = self.forward(input_seq, transfer_source)
        all_item_embeds = self.base_model.item_embed.weight
        scores = torch.matmul(fused_repr, all_item_embeds.T)
        return scores


def init_target_from_source(source: SASRec, target: SASRec):
    with torch.no_grad():
        # positional + final LN
        target.positional_embed.weight.copy_(source.positional_embed.weight)
        target.ln.weight.copy_(source.ln.weight)
        target.ln.bias.copy_(source.ln.bias)
        # blocks
        for b_src, b_tgt in zip(source.blocks, target.blocks):
            # MHAttn
            b_tgt.attn.in_proj_weight.copy_(b_src.attn.in_proj_weight)
            b_tgt.attn.in_proj_bias.copy_(b_src.attn.in_proj_bias)
            b_tgt.attn.out_proj.weight.copy_(b_src.attn.out_proj.weight)
            b_tgt.attn.out_proj.bias.copy_(b_src.attn.out_proj.bias)
            # LayerNorms
            b_tgt.ln1.weight.copy_(b_src.ln1.weight)
            b_tgt.ln1.bias.copy_(b_src.ln1.bias)
            b_tgt.ln2.weight.copy_(b_src.ln2.weight)
            b_tgt.ln2.bias.copy_(b_src.ln2.bias)
            # FFN
            b_tgt.ffn.w1.weight.copy_(b_src.ffn.w1.weight)
            b_tgt.ffn.w1.bias.copy_(b_src.ffn.w1.bias)
            b_tgt.ffn.w2.weight.copy_(b_src.ffn.w2.weight)
            b_tgt.ffn.w2.bias.copy_(b_src.ffn.w2.bias)

# Initialize target model from source model
