import torch
import numpy as np
from tqdm import tqdm


class RLTrainer:
    """Reinforcement Learning fine-tuning for recommendation models."""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device

        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=config.rl_lr
        )

        self.entropy_coeff = config.entropy_coeff
        self.temperature = config.temperature
        self._baseline = 0.0
        self._mom = config.baseline_momentum

        # Feedback to reward mapping
        self.feedback_reward = {
            "click": 0.3, "skip": 0.0
        }

    def events_to_rewards(self, events):
        """Convert feedback events to reward values."""
        rewards = [self.feedback_reward.get(e, 0.0) for e in events]
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)

    def policy_loss(self, logits, actions, rewards):
        """Compute policy gradient loss."""
        logp_all = torch.log_softmax(logits / self.temperature, dim=1)
        probs = torch.exp(logp_all)
        logp_act = logp_all.gather(1, actions.view(-1, 1)).squeeze(1)
        entropy = -(probs * logp_all).sum(dim=1)

        # Update baseline
        with torch.no_grad():
            self._baseline = self._mom * self._baseline + (1 - self._mom) * rewards.mean().item()

        adv = rewards - self._baseline
        loss = -(adv * logp_act + self.entropy_coeff * entropy).mean()

        return loss, {
            "mean_logp": logp_act.mean().item(),
            "entropy": entropy.mean().item(),
            "advantage": adv.mean().item()
        }

    def step_batch(self, batch, sample_actions=True):
        """Process one batch with RL update."""
        self.model.train()

        input_seq = batch["input_seq"].to(self.device)
        pos_items = batch["target"].to(self.device)
        neg_items = batch["neg_items"].to(self.device)
        transfer = batch.get("transfer_src")

        if transfer is not None:
            transfer = transfer.to(self.device)

        # Get logits
        if hasattr(self.model, 'base'):  # Transfer model
            candidates = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)
            fused = self.model(input_seq.long(), transfer_src=transfer)
            cand_emb = self.model.base.item_embed(candidates.long())
            logits = torch.bmm(cand_emb, fused.unsqueeze(-1)).squeeze(-1)
        else:  # Regular model
            candidates = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)
            logits = self.model(input_seq, candidate_items=candidates)

        # Sample or select actions
        if sample_actions:
            probs = torch.softmax(logits / self.temperature, dim=1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            actions = torch.argmax(logits, dim=1)

        # Simulate feedback
        events = ["click" if a.item() == 0 else "skip" for a in actions]
        rewards = self.events_to_rewards(events)

        # Compute loss and update
        loss, stats = self.policy_loss(logits, actions, rewards)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        acc = (actions == 0).float().mean().item()

        torch.save(self.model.state_dict(), f"{self.config.model_dir}/transfer_domain_rl/rl_finetuned_model.pth")

        return {
            "loss": loss.item(),
            "avg_reward": rewards.mean().item(),
            "hit_rate": acc,
            **stats
        }

    def finetune(self, data_loader, epochs=None):
        """Run RL fine-tuning."""
        if epochs is None:
            epochs = self.config.rl_epochs

        history = {
            "loss": [], "avg_reward": [], "hit_rate": [],
            "entropy": [], "advantage": []
        }

        for epoch in range(epochs):
            stats = []

            for batch in tqdm(data_loader, desc=f"  [RL Epoch {epoch + 1}/{epochs}]"):
                s = self.step_batch(batch, sample_actions=True)
                stats.append(s)

            # Average metrics
            m = {k: float(np.mean([x[k] for x in stats])) for k in stats[0].keys()}

            print(f"   Loss: {m['loss']:.4f} | Reward: {m['avg_reward']:.4f} | "
                  f"Hit: {m['hit_rate']:.4f} | Entropy: {m['entropy']:.4f} | Adv: {m['advantage']:.4f}")

            for key in history.keys():
                history[key].append(m[key])

        return history