import torch.nn as nn
import torch
from .mlp import MLP

class FM(nn.Module):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self):
        super(FM, self).__init__()

    def forward(self, inputs):
        fm_input = inputs

        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=1, keepdim=False)

        return cross_term

class DeepFM(nn.Module):
    def __init__(self,
        user_vocab_size: int,
        item_vocab_size: int,
        embedding_dim: int,
        input_user_feature_size: int,
        input_item_feature_size: int):

        super(DeepFM, self).__init__()

        # Embedding table for the user embeddings.
        self.user_emb_table = nn.Embedding(
            num_embeddings=user_vocab_size,
            embedding_dim=embedding_dim,
        )

        # Embedding table for the item embeddings.
        self.item_emb_table = nn.Embedding(
            num_embeddings=item_vocab_size,
            embedding_dim=embedding_dim,
        )

        dense_feature_size = input_user_feature_size + input_item_feature_size

        self.fm = FM()
        self.deep = MLP(input_size=2 * embedding_dim + dense_feature_size,
                        hidden_size=embedding_dim,
                        output_size=1,
                        hidden_layers=3)
        
        

    def forward(self, input):

        user_ids, item_ids, user_features, item_features = input

        user_id_embeddings = self.user_emb_table(user_ids)
        item_id_embeddings = self.item_emb_table(item_ids)

        if len(user_ids.shape) > 1:
            user_id_embeddings = torch.sum(user_id_embeddings, dim=-2)
            item_id_embeddings = torch.sum(item_id_embeddings, dim=-2)

        combined_in = torch.cat([user_id_embeddings, item_id_embeddings, user_features, item_features], dim=-1)

        
        deep_out = self.deep(combined_in).squeeze(1)
        fm_out = self.fm(combined_in)
        logit = deep_out + fm_out
       
        output = torch.sigmoid(logit)

        return output
    


class DeepFMs(nn.Module):
    def __init__(self,
        user_vocab_size: int,
        item_vocab_size: int,
        vocab_sizes: list,
        embedding_dim: int,
        ):

        super(DeepFMs, self).__init__()

        self.embedding_dim = embedding_dim
        self.sparse_feature_size = len(vocab_sizes)



        # Embedding table for the user embeddings.
        self.user_emb_table = nn.Embedding(
            num_embeddings=user_vocab_size,
            embedding_dim=embedding_dim,
        )

        # Embedding table for the item embeddings.
        self.item_emb_table = nn.Embedding(
            num_embeddings=item_vocab_size,
            embedding_dim=embedding_dim,
        )


        # Create a list to hold the embedding tables
        self.embedding_tables = nn.ModuleList([
            nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
            for vocab_size in vocab_sizes
        ])

        self.fm = FM()
        self.deep = MLP(input_size=(self.sparse_feature_size + 2)* embedding_dim,
                        hidden_size=embedding_dim,
                        output_size=1,
                        hidden_layers=3)
      
        

    def forward(self, input):

        user_ids, item_ids, sparse_features = input

        user_id_embeddings = self.user_emb_table(user_ids)
        item_id_embeddings = self.item_emb_table(item_ids)

        if len(user_ids.shape) > 1:
            user_id_embeddings = torch.sum(user_id_embeddings, dim=-2)
            item_id_embeddings = torch.sum(item_id_embeddings, dim=-2)


        sparse_feature_embeddings = torch.concat([
            self.embedding_tables[i](sparse_features[:, i]) for i in range(sparse_features.shape[-1])
        ], dim=-1)

        combined_in = torch.cat([user_id_embeddings, item_id_embeddings, sparse_feature_embeddings], dim=-1)
        
        deep_out = self.deep(combined_in).squeeze(1)
        fm_out = self.fm(combined_in)
        logit = deep_out + fm_out
       
        output = torch.sigmoid(logit)

        return output
    