import torch
import torch.nn as nn
from models.mlp import MLP


class DLRM(nn.Module):
    def __init__(
        self,
        user_vocab_size: int,
        item_vocab_size: int,
        embedding_dim: int,
        input_user_feature_size: int,
        input_item_feature_size: int,
    ):
        super().__init__()

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

        self.bot_user_MLP = MLP(
            input_size=input_user_feature_size,
            hidden_size=embedding_dim * 2,
            output_size=embedding_dim,
        )
        self.bot_item_MLP = MLP(
            input_size=input_item_feature_size,
            hidden_size=embedding_dim * 2,
            output_size=embedding_dim,
        )
        self.top_MLP = MLP(
            input_size=embedding_dim * 4, hidden_size=embedding_dim * 2, output_size=1
        )

    def forward(self, input):
        user_ids, item_ids, user_features, item_features = input

        user_id_embeddings = self.user_emb_table(user_ids)
        item_id_embeddings = self.item_emb_table(item_ids)

        if len(user_ids.shape) > 1:
            user_id_embeddings = torch.sum(user_id_embeddings, dim=-2)
            item_id_embeddings = torch.sum(item_id_embeddings, dim=-2)

        user_feature_embeddings = self.bot_user_MLP(user_features)
        item_feature_embeddings = self.bot_item_MLP(item_features)

        concat = torch.concat(
            [
                user_id_embeddings,
                item_id_embeddings,
                user_feature_embeddings,
                item_feature_embeddings,
            ],
            dim=-1,
        )

        raw = self.top_MLP(concat)

        output = torch.sigmoid(raw)

        return output


class DLRMxL(nn.Module):
    def __init__(
        self,
        user_vocab_size: int,
        item_vocab_size: int,
        tag_vocab_size: int,
        embedding_dim: int,
        dense_feature_size: int,
    ):
        super().__init__()
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

        # Create a list to hold the embedding tables
        self.tag_emb_table = nn.Embedding(
            num_embeddings=tag_vocab_size,
            embedding_dim=embedding_dim,
        )

        self.bot_MLP = MLP(
            input_size=dense_feature_size,
            hidden_size=embedding_dim * 2,
            output_size=embedding_dim,
        )
        self.top_MLP = MLP(
            input_size=embedding_dim * 4, hidden_size=embedding_dim, output_size=1
        )

    def forward(self, input):
        user_ids, item_ids, dense_features, sparse_features = input

        user_id_embeddings = self.user_emb_table(user_ids)
        item_id_embeddings = self.item_emb_table(item_ids)

        if len(user_ids.shape) > 1:
            user_id_embeddings = torch.sum(user_id_embeddings, dim=-2)
            item_id_embeddings = torch.sum(item_id_embeddings, dim=-2)

        tag_embeddings = torch.sum(
            torch.stack(
                [
                    self.tag_emb_table(sparse_features[:, i])
                    for i in range(sparse_features.shape[-1])
                ],
                dim=1,
            ),
            dim=1,
        )

        dense_feature_embeddings = self.bot_MLP(dense_features)

        concat = torch.concat(
            [
                user_id_embeddings,
                item_id_embeddings,
                tag_embeddings,
                dense_feature_embeddings,
            ],
            dim=-1,
        )

        raw = self.top_MLP(concat)

        output = torch.sigmoid(raw)

        return output


class DLRMs(nn.Module):
    def __init__(
        self,
        user_vocab_size: int,
        item_vocab_size: int,
        vocab_sizes: list,
        embedding_dim: int,
    ):
        super().__init__()

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
        self.embedding_tables = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
                for vocab_size in vocab_sizes
            ]
        )

        self.top_MLP = MLP(
            input_size=embedding_dim * (self.sparse_feature_size + 2),
            hidden_size=embedding_dim * 2,
            output_size=1,
        )

    def forward(self, input):
        user_ids, item_ids, sparse_features = input

        user_id_embeddings = self.user_emb_table(user_ids)
        item_id_embeddings = self.item_emb_table(item_ids)

        if len(user_ids.shape) > 1:
            user_id_embeddings = torch.sum(user_id_embeddings, dim=-2)
            item_id_embeddings = torch.sum(item_id_embeddings, dim=-2)

        sparse_feature_embeddings = torch.concat(
            [
                self.embedding_tables[i](sparse_features[:, i])
                for i in range(sparse_features.shape[-1])
            ],
            dim=-1,
        )

        concat = torch.concat(
            [user_id_embeddings, item_id_embeddings, sparse_feature_embeddings], dim=-1
        )

        raw = self.top_MLP(concat)

        output = torch.sigmoid(raw)

        return output
