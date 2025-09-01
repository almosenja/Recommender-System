import random
import torch
from torch.utils.data import Dataset

class SASRecDataset(Dataset):
    def __init__(self, data, num_items, max_seq_len=50, pos_items_by_user=None, mode="train", neg_samples=1):
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.neg_samples = neg_samples
        self.all_pos = pos_items_by_user

        self.samples = []
        if mode == "train":
            for user, seq in data.items():
                for i in range(1, len(seq)):
                    self.samples.append({
                        "user": user,
                        "input_seq": seq[:i],
                        "target": seq[i],
                        "full_seq": seq # For negative sampling
                    })
        else:
            for user, (seq, target) in data.items():
                self.samples.append({
                    "user": user,
                    "input_seq": seq,
                    "target": target,
                    "full_seq": seq + [target]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        user = sample["user"]
        seq = sample["input_seq"]
        target = sample["target"]

        # Truncate sequence if > max length
        if len(seq) > self.max_seq_len:
            seq = seq[-self.max_seq_len:]

        # Left-pad sequence with zeros
        pad_len = self.max_seq_len - len(seq)
        padded_seq = [0] * pad_len + seq

        # Negative sampling
        forbid = self.all_pos[user] if self.all_pos is not None else set(sample["full_seq"])
        neg_items = set()

        while len(neg_items) < self.neg_samples:
            neg = random.randint(1, self.num_items - 1)
            if neg not in forbid:
                neg_items.add(neg)

        return {
            "user": sample["user"],
            "input_seq": torch.tensor(padded_seq, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.long),
            "neg_items": torch.tensor(list(neg_items), dtype=torch.long)
        }

class TransferDataset(SASRecDataset):
    # def __init__(self,
    #              data,
    #              num_items,
    #              transfer_matrix,
    #              max_seq_len=50,
    #              mode="train",
    #              neg_samples=1):
    def __init__(self, *args, transfer_matrix=None, **kwargs):

        # super().__init__(data,
        #                  num_items,
        #                  max_seq_len=max_seq_len,
        #                  pos_items_by_user=None,
        #                  mode=mode,
        #                  neg_samples=neg_samples)

        super().__init__(*args, **kwargs)
        self.transfer_matrix = transfer_matrix

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        if self.transfer_matrix is not None:
            user_id = out["user"]
            out["transfer_src"] = self.transfer_matrix[user_id].float()
        return out