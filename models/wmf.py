import torch
import torch.nn as nn
import torch.nn.functional as F


class WMF(nn.Module):
    """Weighted Matrix Factorization Model"""

    def __init__(
        self,
        user_vocab_size: int,
        item_vocab_size: int,
        embedding_dim: int,
        alpha: float = 1.0,
    ):
        """
        Args:
            user_vocab_size (int): Number of unique users.
            item_vocab_size (int): Number of unique items.
            embedding_dim (int): Dimensionality of embeddings.
            alpha (float): Scaling factor for the confidence weights.
        """
        super(WMF, self).__init__()
        self.embedding_dim = embedding_dim
        self.alpha = alpha

        # Embedding tables for user and item embeddings.
        self.user_emb_table = nn.Embedding(
            num_embeddings=user_vocab_size,
            embedding_dim=embedding_dim,
        )
        self.item_emb_table = nn.Embedding(
            num_embeddings=item_vocab_size,
            embedding_dim=embedding_dim,
        )

        self._init_weight_()

    def _init_weight_(self):
        nn.init.xavier_uniform_(self.user_emb_table.weight)
        nn.init.xavier_uniform_(self.item_emb_table.weight)

    def forward(self, user_id, item_id, interaction_matrix, confidence_matrix):
        """
        Compute the reconstruction loss for the interaction matrix.

        Args:
            user_id (Tensor): Tensor of user IDs.
            item_id (Tensor): Tensor of item IDs.
            interaction_matrix (Tensor): Ground truth matrix of interactions.
            confidence_matrix (Tensor): Confidence weights for the interactions.

        Returns:
            loss (Tensor): Weighted matrix factorization loss.
        """
        # Get user and item embeddings
        user_embeddings = self.user_emb_table(user_id)  # [num_users, embedding_dim]
        item_embeddings = self.item_emb_table(item_id)  # [num_items, embedding_dim]

        # Reconstructed interaction matrix
        reconstruction = user_embeddings @ item_embeddings.t()  # [num_users, num_items]

        # Compute loss: Weighted squared error
        interaction_diff = interaction_matrix - reconstruction
        weighted_diff = confidence_matrix * (interaction_diff**2)
        loss = torch.sum(weighted_diff)

        return loss

    def get_scores(self, user_id, item_id):
        """
        Compute scores for user-item pairs.

        Args:
            user_id (Tensor): Tensor of user IDs.
            item_id (Tensor): Tensor of item IDs.

        Returns:
            scores (Tensor): Predicted interaction scores.
        """
        user_embeddings = self.user_emb_table(user_id)  # [num_users, embedding_dim]
        item_embeddings = self.item_emb_table(item_id)  # [num_items, embedding_dim]

        scores = user_embeddings @ item_embeddings.t()  # [num_users, num_items]
        return scores
