import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):

    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss
    




class DAULoss(nn.Module):
    """DirectAU loss proposed in https://arxiv.org/abs/2206.12811

    Parameter
    ---------
    gamma: float
        the trade-off value between alignment and uniform
    """

    def __init__(self, gamma=1.0):
        super(DAULoss, self).__init__()
        self.gamma = gamma

    def forward(self, user_embedding, positive_item_embedding):
        """Get the numerical value of the BPR loss function,
            Implementation based on: https://github.com/THUwangcy/DirectAU

        Parameters
        ----------
        model_output: TripletModelOutput
            model_output.user_embedding: torch.Tensor:
                user embedding matrix (batch_size x embedding_dim)
            model_output.positive_item_embedding: torch.Tensor,
                positive item embedding matrix (batch_size x embedding_dim)
            model_output.negative_item_embedding:
                negative item embedding matrix (batch_size x embedding_dim)
        """

        align = self.alignment(user_embedding, positive_item_embedding)
        uniform = (
            self.uniformity(user_embedding) + self.uniformity(positive_item_embedding)
        ) / 2
        return align + self.gamma * uniform

    @staticmethod
    def alignment(
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    @staticmethod
    def uniformity(
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()



class iALSLoss(nn.Module):
    def __init__(self,  alpha=0.1):
        """
        Initializes the iALS loss module for sparse positive-only interaction matrices.

        Args:
        - reg_lambda (float): Regularization parameter.
        - alpha (float): Scaling factor for confidence.
        """
        super(iALSLoss, self).__init__()
        self.alpha = alpha

    def forward(self, user_embeddings, positive_item_embeddings, negative_item_embeddings):
        """
        Computes the iALS loss with sparse interactions.

        Args:
        - user_factors (torch.Tensor): User latent factor matrix of shape (num_users, latent_dim).
        - item_factors (torch.Tensor): Item latent factor matrix of shape (num_items, latent_dim).
        - interactions (scipy.sparse.csr_matrix): Positive-only interaction matrix (binary, sparse).

        Returns:
        - loss (torch.Tensor): Scalar loss value.
        """

        # Confidence for positive interactions
        confidence = 1 + self.alpha  # Since all entries are 1 in sparse form

        positive_predictions = (user_embeddings * positive_item_embeddings).sum(dim=1)

        # Compute loss for positive interactions
        positive_loss = confidence * (1 - positive_predictions) ** 2  # Binary preference = 1

        # Compute loss for negative (unobserved) interactions
        # Approximate negative loss using all interactions

        negative_predictions = (user_embeddings * negative_item_embeddings).sum(dim=1)
    
        negative_loss = (negative_predictions ** 2).sum()  # Exclude positives

        

        # Total loss
        loss = positive_loss.sum() + negative_loss 

        return loss

