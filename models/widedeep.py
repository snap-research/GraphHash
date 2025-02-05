import torch
import torch.nn as nn
from .mlp import MLP


class Wide(nn.Module):
    def __init__(self, feature_dim, embedding_dim):
        super(Wide, self).__init__()
        self.wide = nn.Linear(feature_dim, embedding_dim)

    def forward(self, x):
        x = self.wide(x)
        return x


class WideDeep(nn.Module):
    def __init__(
        self,
        user_vocab_size: int,
        item_vocab_size: int,
        embedding_dim: int,
        input_user_feature_size: int,
        input_item_feature_size: int,
    ):
        super().__init__()

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

        # Create a list to hold the embedding tables

        self.wide = Wide(
            feature_dim=2 * embedding_dim + dense_feature_size,
            embedding_dim=embedding_dim,
        )
        self.deep = MLP(
            input_size=2 * embedding_dim + dense_feature_size,
            hidden_size=embedding_dim * 2,
            output_size=embedding_dim,
            hidden_layers=3,
        )
        self.together = nn.Linear(2 * embedding_dim, 1)

    def forward(self, input):
        user_ids, item_ids, user_features, item_features = input

        user_id_embeddings = self.user_emb_table(user_ids)
        item_id_embeddings = self.item_emb_table(item_ids)

        if len(user_ids.shape) > 1:
            user_id_embeddings = torch.sum(user_id_embeddings, dim=-2)
            item_id_embeddings = torch.sum(item_id_embeddings, dim=-2)

        combined_in = torch.cat(
            [user_id_embeddings, item_id_embeddings, user_features, item_features],
            dim=-1,
        )
        wide_out = self.wide(combined_in)
        deep_out = self.deep(combined_in)

        logit = self.together(torch.cat([wide_out, deep_out], dim=-1))

        output = torch.sigmoid(logit)

        return output


class WideDeepxL(nn.Module):
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

        # Create a list to hold the embedding tables

        self.wide = Wide(
            feature_dim=3 * embedding_dim + dense_feature_size,
            embedding_dim=embedding_dim,
        )
        self.deep = MLP(
            input_size=3 * embedding_dim + dense_feature_size,
            hidden_size=embedding_dim * 2,
            output_size=embedding_dim,
            hidden_layers=3,
        )
        self.together = nn.Linear(2 * embedding_dim, 1)

    def forward(self, input):
        user_hashed_ids, item_hashed_ids, dense_features, sparse_features = input

        user_id_embeddings = self.user_emb_table(user_hashed_ids)
        item_id_embeddings = self.item_emb_table(item_hashed_ids)

        if len(user_hashed_ids.shape) > 1:
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

        combined_in = torch.cat(
            [user_id_embeddings, item_id_embeddings, tag_embeddings, dense_features],
            dim=-1,
        )
        wide_out = self.wide(combined_in)
        deep_out = self.deep(combined_in)

        logit = self.together(torch.cat([wide_out, deep_out], dim=-1))

        output = torch.sigmoid(logit)

        return output


class WideDeeps(nn.Module):
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

        # Create a list to hold the embedding tables

        self.wide = Wide(
            feature_dim=(self.sparse_feature_size + 2) * embedding_dim,
            embedding_dim=embedding_dim,
        )
        self.deep = MLP(
            input_size=(self.sparse_feature_size + 2) * embedding_dim,
            hidden_size=embedding_dim * 2,
            output_size=embedding_dim,
            hidden_layers=3,
        )
        self.together = nn.Linear(2 * embedding_dim, 1)

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

        combined_in = torch.concat(
            [user_id_embeddings, item_id_embeddings, sparse_feature_embeddings], dim=-1
        )

        wide_out = self.wide(combined_in)
        deep_out = self.deep(combined_in)

        logit = self.together(torch.cat([wide_out, deep_out], dim=-1))

        output = torch.sigmoid(logit)

        return output
