import torch
import torch.nn as nn
from .baserec import BaseRecommender
import scipy as sp
import numpy as np


class LightGCN(BaseRecommender):
    def __init__(
        self,
        user_vocab_size: int,
        item_vocab_size: int,
        embedding_dim: int,
        user_hashed_ids,
        item_hashed_ids,
        biadjacency=sp.sparse.csr_matrix(np.ones((3, 3))),
        hash_type="full",
        num_layers=2,
    ):
        super(LightGCN, self).__init__()
        self.embedding_dim = embedding_dim
        # define layers
        self.user_emb_table = nn.Embedding(user_vocab_size, embedding_dim)
        self.item_emb_table = nn.Embedding(item_vocab_size, embedding_dim)

        self.interaction_matrix = biadjacency
        self.n_users = biadjacency.shape[0]
        self.n_items = biadjacency.shape[1]
        self.n_layers = num_layers

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.user_hashed_ids = torch.tensor(user_hashed_ids).to(device)
        self.item_hashed_ids = torch.tensor(item_hashed_ids).to(device)

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(device)

        self.hash_type = hash_type

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """

        # build adj matrix
        A = sp.sparse.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.T

        row_indices, col_indices = inter_M.nonzero()
        row_indices_t, col_indices_t = inter_M_t.nonzero()

        data_dict = dict(
            zip(zip(row_indices, col_indices + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(row_indices_t + self.n_users, col_indices_t),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.sparse.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.sparse.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))

        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        if self.hash_type not in ["double", "double_frequency", "double_graph"]:
            user_embeddings = self.user_emb_table(self.user_hashed_ids)
            item_embeddings = self.item_emb_table(self.item_hashed_ids)

        else:
            user_embeddings = self.user_emb_table(self.user_hashed_ids)
            item_embeddings = self.item_emb_table(self.item_hashed_ids)
            user_embeddings = torch.sum(user_embeddings, dim=-2)
            item_embeddings = torch.sum(item_embeddings, dim=-2)

        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)

        return ego_embeddings

    def propagate(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def forward(self, user_id, pos_item_id, neg_item_id):
        user_all_embeddings, item_all_embeddings = self.propagate()
        user_id_embeddings = user_all_embeddings[user_id]
        pos_item_id_embeddings = item_all_embeddings[pos_item_id]
        neg_item_id_embeddings = item_all_embeddings[neg_item_id]

        pos_score = torch.sum(user_id_embeddings * pos_item_id_embeddings, dim=-1)

        neg_score = torch.sum(user_id_embeddings * neg_item_id_embeddings, dim=-1)

        return pos_score, neg_score

    def get_scores(self, hash_type, device, user_id, item_id):
        user_all_embeddings, item_all_embeddings = self.propagate()
        user_id_embeddings = user_all_embeddings[user_id]
        item_id_embeddings = item_all_embeddings[item_id]

        scores = user_id_embeddings @ item_id_embeddings.t()

        return scores
