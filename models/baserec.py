import torch
import numpy as np
from abc import ABC, abstractmethod
import torch.nn as nn


class BaseRecommender(ABC, nn.Module):
    """Base class for recommender models."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Abstract method to be implemented by subclasses for forward pass."""
        pass

    @abstractmethod
    def get_scores(self, *args, **kwargs):
        """Abstract method to be implemented by subclasses for getting the whole scores."""
        pass

    def recommend(
        self,
        hash_type,
        train_candidates,
        test_candidates,
        user_clusters,
        item_clusters,
        device,
        batch_size=20000,
        k=10,
    ):
        """Recommends top-k items for users."""

        top_k_indices_all = []

        test_users = list(test_candidates.keys())
        num_batches = np.ceil(len(test_users) / batch_size)
        test_user_batches = np.array_split(test_users, num_batches)

        for user_id_batch in test_user_batches:
            user_id_hashed = user_clusters[user_id_batch]
            all_item_id_hashed = item_clusters

            scores = self.get_scores(
                hash_type, device, user_id_hashed, all_item_id_hashed
            )

            mask_user_idx = []
            mask_item_idx = []
            user_id_batch_2_idx = {user: idx for idx, user in enumerate(user_id_batch)}
            for user in user_id_batch:
                mask_user_idx.extend(
                    [user_id_batch_2_idx[user]] * len(train_candidates[user])
                )
                mask_item_idx.extend(train_candidates[user])
            mask_user_idx = torch.tensor(mask_user_idx, dtype=torch.int64).to(device)
            mask_item_idx = torch.tensor(mask_item_idx, dtype=torch.int64).to(device)
            scores[mask_user_idx, mask_item_idx] = -1e4

            _, top_k_indices = torch.topk(scores, k, dim=1)
            top_k_indices_all.append(top_k_indices)

        top_k_indices_all = torch.cat(top_k_indices_all, dim=0)

        return top_k_indices_all
