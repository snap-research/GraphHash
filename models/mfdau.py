import torch
import torch.nn as nn
from .baserec import BaseRecommender

class MFDAU(BaseRecommender):
    """Simple model (MF)"""

    def __init__(
        self,
        user_vocab_size: int,
        item_vocab_size: int,
        embedding_dim: int,
    ):

        super(MFDAU, self).__init__()
        self.embedding_dim = embedding_dim

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

        self._init_weight_()




   

    def _init_weight_(self):
        nn.init.xavier_uniform_(self.user_emb_table.weight)
        nn.init.xavier_uniform_(self.item_emb_table.weight)


    def forward(self, user_id, pos_item_id):
       
        user_id_embeddings = self.user_emb_table(user_id)
        pos_item_id_embeddings = self.item_emb_table(pos_item_id)
        

        if len(user_id.shape) > 1:

            user_id_embeddings = torch.sum(user_id_embeddings, dim=-2)
            pos_item_id_embeddings = torch.sum(pos_item_id_embeddings, dim=-2)
        


        return user_id_embeddings, pos_item_id_embeddings
    
    def get_scores(self, hash_type, device, user_id_hashed, item_id_hashed):
        if len(user_id_hashed.shape) == 1 or hash_type == 'dhe':

            user_id_embeddings = self.user_emb_table(user_id_hashed)
            item_id_embeddings = self.item_emb_table(item_id_hashed)

        else: 
            user_id_embeddings = self.user_emb_table(user_id_hashed)
            item_id_embeddings = self.item_emb_table(item_id_hashed)

            user_id_embeddings = torch.sum(user_id_embeddings, dim=-2)
            item_id_embeddings = torch.sum(item_id_embeddings, dim=-2)



        scores = user_id_embeddings @ item_id_embeddings.t()
        
        return scores


    
   