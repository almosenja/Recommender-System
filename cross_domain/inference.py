import torch
import numpy as np
from typing import List, Dict, Optional


class RecommendationInference:
    def __init__(self, model, data_processor, config):
        self.model = model
        self.data_processor = data_processor
        self.config = config
        self.device = config.device

    @torch.no_grad()
    def get_recommendations(self, user_sequence: List[int], k: int = 10,
                            transfer_vec: Optional[np.ndarray] = None) -> Dict:
        """Get top-k recommendations for a user sequence."""
        self.model.eval()
        self.model.to(self.device)

        # Prepare input
        seq = user_sequence[-self.config.max_seq_len:]
        pad_len = self.config.max_seq_len - len(seq)
        padded_seq = [0] * pad_len + seq
        input_seq = torch.tensor([padded_seq], dtype=torch.long, device=self.device)

        # Get scores
        if transfer_vec is not None and hasattr(self.model, 'predict_next'):
            transfer = torch.tensor(transfer_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            scores = self.model.predict_next(input_seq, transfer_src=transfer)
        else:
            scores = self.model.predict_next(input_seq)

        # Get top-k items
        scores = scores.squeeze(0)
        scores[0] = -float('inf')  # Exclude padding

        # Exclude items already in sequence
        for item in user_sequence:
            if item < len(scores):
                scores[item] = -float('inf')

        top_scores, top_items = torch.topk(scores, k)

        return {
            "items": top_items.cpu().numpy(),
            "scores": top_scores.cpu().numpy()
        }

    def display_recommendations(self, user_raw: str, user_sequences: Dict,
                                k: int = 10, metadata: Optional[Dict] = None):
        """Display recommendations for a user in a formatted way."""
        # Get user ID
        if self.data_processor.user_encoder:
            try:
                user_id = self.data_processor.user_encoder.transform([user_raw])[0]
            except:
                print(f"User {user_raw} not found in encoder")
                return
        else:
            user_id = int(user_raw)

        if user_id not in user_sequences:
            print(f"User {user_raw} (ID: {user_id}) has no interaction history")
            return

        # Get user sequence
        user_seq = user_sequences[user_id]

        # Get recommendations
        recs = self.get_recommendations(user_seq, k=k)

        # Display results
        print(f"\nUser: {user_raw} (ID: {user_id})")
        print("-" * 80)

        # Show recent interactions
        print("Recent interactions (most recent first):")
        recent_items = user_seq[-5:][::-1]  # Last 5 items, reversed
        for item_id in recent_items:
            if self.data_processor.item_encoder:
                item_raw = self.data_processor.item_encoder.inverse_transform([item_id - 1])[0]  # -1 for shift
            else:
                item_raw = str(item_id)

            item_info = f"  - {item_raw} (ID: {item_id})"
            if metadata and item_raw in metadata:
                item_info += f" - {metadata[item_raw].get('title', '')[:50]}"
            print(item_info)

        # Show recommendations
        print(f"\nTop {k} Recommendations:")
        for i, (item_id, score) in enumerate(zip(recs['items'], recs['scores']), 1):
            if self.data_processor.item_encoder:
                item_raw = self.data_processor.item_encoder.inverse_transform([item_id - 1])[0]
            else:
                item_raw = str(item_id)

            rec_info = f"  {i}. {item_raw} (ID: {item_id}, Score: {score:.4f})"
            if metadata and item_raw in metadata:
                rec_info += f" - {metadata[item_raw].get('title', '')[:50]}"
            print(rec_info)