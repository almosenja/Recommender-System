import torch, numpy as np

FEEDBACK_REWARD = {
    "like": 1.0, "purchase": 1.0, "click": 0.3,
    "add_to_cart": 0.7, "dislike": -0.5, "skip": 0.0
}

def events_to_rewards(events, device="cpu"):
    reward = [FEEDBACK_REWARD.get(event, 0.0) for event in events]
    return torch.tensor(reward, dtype=torch.float32, device=device)

class PolicyGradientTrainer:
    def __init__(self, model, lr=5e-5, entropy_coeff=0.01, temperature=1.0, baseline_momentum=0.9, device="cpu"):
        self.model = model.to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.entropy_coeff = entropy_coeff
        self.temperature = temperature
        self.device = device
        self._baseline = 0.0
        self._mom = baseline_momentum

    def _scores(self, input_seq, xfer_src, candidates):
        candidates = candidates.long(); input_seq = input_seq.long()
        fused = self.model(input_seq, transfer_src=xfer_src)                   # [B, D]
        cand_emb = self.model.base.item_embed(candidates)                      # [B, N, D]
        scores = torch.bmm(cand_emb, fused.unsqueeze(-1)).squeeze(-1)          # [B, N]
        return scores

    def _policy_loss(self, logits, actions, rewards):
        logp_all = torch.log_softmax(logits / self.temperature, dim=1)
        probs = torch.exp(logp_all)
        logp_act = logp_all.gather(1, actions.view(-1,1)).squeeze(1)
        entropy = -(probs * logp_all).sum(dim=1)

        with torch.no_grad():
            self._baseline = self._mom * self._baseline + (1 - self._mom) * rewards.mean().item()
        adv = rewards - self._baseline
        loss = -(adv * logp_act + self.entropy_coeff * entropy).mean()
        return loss, logp_act.mean().item(), entropy.mean().item(), adv.mean().item()

    def step_offline_demo_batch(self, batch, sample_actions=True):
        self.model.train()
        inp = batch["input_seq"].to(self.device)
        tgt = batch["target"].to(self.device)
        neg = batch["neg_items"].to(self.device)
        xfer = batch["transfer_src"].to(self.device)

        candidates = torch.cat([tgt.unsqueeze(1), neg], dim=1)
        logits = self._scores(inp, xfer, candidates)

        if sample_actions:
            probs = torch.softmax(logits / self.temperature, dim=1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            actions = torch.argmax(logits, dim=1)

        events = ["click" if a.item() == 0 else "skip" for a in actions]
        rewards = events_to_rewards(events, device=self.device)

        loss, mean_logp, mean_H, mean_adv = self._policy_loss(logits, actions, rewards)
        self.opt.zero_grad(); loss.backward(); self.opt.step()

        acc = (actions == 0).float().mean().item()
        return {"loss": loss.item(), "avg_reward": rewards.mean().item(),
                "hit_rate": acc, "mean_logp": mean_logp, "entropy": mean_H, "adv": mean_adv}

    def offline_demo_finetune(self, loader, steps=1, sample_actions=True):
        for ep in range(steps):
            stats = []
            for batch in loader:
                s = self.step_offline_demo_batch(batch, sample_actions=sample_actions)
                stats.append(s)
            m = {k: float(np.mean([x[k] for x in stats])) for k in stats[0].keys()}
            print(f"[RL demo] Epoch {ep+1}/{steps}  loss {m['loss']:.4f}  avgR {m['avg_reward']:.3f}  "
                  f"hit {m['hit_rate']:.3f}  H {m['entropy']:.3f}  adv {m['adv']:.3f}")
