import torch
import torch.nn as nn
from .mlp import MLP
from .baserec import BaseRecommender


class NeuMF(BaseRecommender):
    def __init__(
        self,
        user_vocab_size: int,
        item_vocab_size: int,
        embedding_dim: int,
    ):
        self.embedding_dim = embedding_dim

        # Call the superclass __init__ method first
        super(NeuMF, self).__init__()

        # define layers
        self.user_emb_table = nn.Embedding(user_vocab_size, embedding_dim)
        self.item_emb_table = nn.Embedding(item_vocab_size, embedding_dim)
        self.user_mlp_embedding = nn.Embedding(user_vocab_size, embedding_dim)
        self.item_mlp_embedding = nn.Embedding(item_vocab_size, embedding_dim)
        self.mlp_layers = MLP(input_size=2 * embedding_dim, hidden_size=embedding_dim, output_size=embedding_dim)
        self.predict_layer = nn.Linear(2 * embedding_dim, 1)


    def forward(self, user, item, neg_item):
        user_mf_e = self.user_emb_table(user)
        item_mf_e = self.item_emb_table(item)
        neg_item_mf_e = self.item_emb_table(neg_item)

        user_mlp_e = self.user_mlp_embedding(user)
        item_mlp_e = self.item_mlp_embedding(item)
        neg_item_mlp_e = self.item_mlp_embedding(neg_item)

        if len(user.shape) > 1:

            user_mf_e = torch.sum(user_mf_e, dim=-2)
            item_mf_e = torch.sum(item_mf_e, dim=-2)
            neg_item_mf_e = torch.sum(neg_item_mf_e, dim=-2)

            user_mlp_e = torch.sum(user_mlp_e, dim=-2)
            item_mlp_e = torch.sum(item_mlp_e, dim=-2)
            neg_item_mlp_e = torch.sum(neg_item_mlp_e, dim=-2)





        mf_output = torch.mul(user_mf_e, item_mf_e)  
        neg_mf_output = torch.mul(user_mf_e, neg_item_mf_e)  


        mlp_output = self.mlp_layers(torch.cat((user_mlp_e, item_mlp_e), -1))  
        neg_mlp_output = self.mlp_layers(torch.cat((user_mlp_e, neg_item_mlp_e), -1))  
    


        pos_score = self.predict_layer(torch.cat((mf_output, mlp_output), -1))
        neg_score = self.predict_layer(torch.cat((neg_mf_output, neg_mlp_output), -1))

        return pos_score, neg_score
    

    def get_scores(self, hash_type, device, user_id_hashed, item_id_hashed):
        if len(user_id_hashed.shape) == 1 or hash_type == 'dhe':

            user_id_embeddings = self.user_emb_table(user_id_hashed)
            item_id_embeddings = self.item_emb_table(item_id_hashed)
            user_mlp_e = self.user_mlp_embedding(user_id_hashed)
            item_mlp_e = self.item_mlp_embedding(item_id_hashed)

        else:
            user_id_embeddings = self.user_emb_table(user_id_hashed)
            item_id_embeddings = self.item_emb_table(item_id_hashed)
            user_mlp_e = self.user_mlp_embedding(user_id_hashed)
            item_mlp_e = self.item_mlp_embedding(item_id_hashed)

            user_id_embeddings = torch.sum(user_id_embeddings, dim=-2)
            item_id_embeddings = torch.sum(item_id_embeddings, dim=-2)
            user_mlp_e = torch.sum(user_mlp_e, dim=-2)
            item_mlp_e = torch.sum(item_mlp_e, dim=-2)
           

        mf_output = user_id_embeddings.unsqueeze(1) * item_id_embeddings.unsqueeze(0) # shape: (# users, # items, embed_dim)
        mf_output = mf_output.view(-1, self.embedding_dim)

       


        # Expand dimensions for broadcasting
        user_expanded = user_mlp_e.unsqueeze(1).expand(user_mlp_e.shape[0], item_mlp_e.shape[0], self.embedding_dim)  # Shape (N, M, d)
        item_expanded = item_mlp_e.unsqueeze(0).expand(user_mlp_e.shape[0], item_mlp_e.shape[0], self.embedding_dim)  # Shape (N, M, d)

        # Concatenate user and item embeddings
        pairwise_features = torch.cat((user_expanded, item_expanded), dim=2)  # Shape (N, M, 2d)

        # Reshape to (N * M, 2d) to process through the MLP in a single batch
        pairwise_features_reshaped = pairwise_features.view(-1, 2 * self.embedding_dim)  # Shape (N * M, 2d)

        # Forward pass through the MLP
        mlp_output = self.mlp_layers(pairwise_features_reshaped)  # Shape (N * M, d)



        scores = self.predict_layer(torch.cat((mf_output, mlp_output), -1)) # Shape (N * M, 1)

        scores = scores.view(user_id_hashed.shape[0], item_id_hashed.shape[0])



        return scores