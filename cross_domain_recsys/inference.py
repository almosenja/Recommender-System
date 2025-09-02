import torch
from typing import List, Dict, Optional

class RecommendationInference:
    def __init__(self, model, data_processor, config, transfer_matrix):
        self.transfer_matrix = transfer_matrix
        self.model = model
        self.data_processor = data_processor
        self.config = config
        self.device = config.device

    @torch.no_grad()
    def get_recommendations(self, user_sequence: List[int],  user_id: int, k: int = 10) -> Dict:
        """Get top-k recommendations for a user sequence."""
        self.model.eval()
        self.model.to(self.device)

        # Prepare input
        seq = user_sequence[-self.config.max_seq_len:]
        pad_len = self.config.max_seq_len - len(seq)
        padded_seq = [0] * pad_len + seq
        input_seq = torch.tensor([padded_seq], dtype=torch.long, device=self.device)

        # Get transfer vector for this user
        transfer_vec = None
        if user_id is not None and user_id < len(self.transfer_matrix):
            user_transfer_vec = self.transfer_matrix[user_id].unsqueeze(0).to(self.device)
            if user_transfer_vec.norm() > 0:
                transfer_vec = user_transfer_vec

        # Get model predictions and squeeze the batch dimension
        logits = self.model.predict_next(input_seq, transfer_src=transfer_vec).squeeze(0) # <-- FIX 1: SQUEEZE HERE

        # Exclude padding item (ID 0) and items in the user's history
        logits[0] = -float("inf")
        if user_sequence:
            history_items = torch.tensor(user_sequence, dtype=torch.long, device=self.device)
            logits.index_fill_(0, history_items, -float("inf"))

        # Get top-k items
        top_scores, top_items = torch.topk(logits, k)
        return {
            "items": top_items.cpu().numpy(),
            "scores": top_scores.cpu().numpy(),
            "has_transfer": transfer_vec is not None
        }

    def display_recommendations(self, user_raw: str, user_sequences: Dict,
                                k: int = 10, metadata: Optional[Dict] = None):
        """Display recommendations for a user in a formatted way."""
        if self.data_processor.user_encoder:
            try:
                user_id = self.data_processor.user_encoder.transform([user_raw])[0]
            except ValueError:
                print(f"User {user_raw} not found in encoder")
                return
        else:
            user_id = int(user_raw)

        if user_id not in user_sequences:
            print(f"User {user_raw} (ID: {user_id}) has no interaction history")
            return

        user_seq = user_sequences[user_id]
        recs = self.get_recommendations(user_seq, k=k, user_id=user_id)

        print(f"\nUser: {user_raw} (ID: {user_id})")
        print(f"Transfer status: {'Has source domain information' if recs.get('has_transfer') else 'No source domain information'}")
        print("-" * 80)

        print("Recent interactions (most recent first):")
        recent_items = user_seq[-5:][::-1]
        for item_id in recent_items:
            try:
                item_raw = self.data_processor.item_encoder.inverse_transform([item_id - 1])[0]
                item_info = f"  - {item_raw} (ID: {item_id})"
                if metadata and item_raw in metadata:
                    item_info += f" - {metadata[item_raw][:50]}"
                print(item_info)
            except IndexError:
                print(f"  - Unknown Item (ID: {item_id})")


        print(f"\nTop {k} Recommendations:")
        for i, (item_id, score) in enumerate(zip(recs['items'], recs['scores']), 1):
            try:
                item_raw = self.data_processor.item_encoder.inverse_transform([item_id - 1])[0]
                rec_info = f"  {i}. {item_raw} (ID: {item_id}, Score: {score:.4f})"
                if metadata and item_raw in metadata:
                    rec_info += f" - {metadata[item_raw][:50]}"
                print(rec_info)
            except IndexError:
                 print(f"  {i}. Unknown Item (ID: {item_id}, Score: {score:.4f})")