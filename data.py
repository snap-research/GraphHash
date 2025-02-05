import torch
from torch.utils.data import Dataset
import numpy as np
import h5py


class UserItemRatingDataset(Dataset):
    def __init__(self, edge_index):
        self.edge_index = edge_index

        # Extract the unique user and item IDs from the edge_index
        self.user_ids = np.unique(edge_index[0])
        self.item_ids = np.unique(edge_index[1])
        self.num_users = len(self.user_ids)
        self.num_items = len(self.item_ids)

    def __len__(self):
        return self.edge_index.shape[-1]

    def __getitem__(self, idx):
        user_id = self.edge_index[0, idx].item()
        item_id = self.edge_index[1, idx].item()

        return (user_id, item_id)


class UserItemRatingDatasetv2(Dataset):
    def __init__(self, edge_index, user_features, item_features, ratings):
        self.edge_index = edge_index
        self.user_features = user_features
        self.item_features = item_features
        self.ratings = ratings

        # Extract the unique user and item IDs from the edge_index
        self.user_ids = np.unique(edge_index[0])
        self.item_ids = np.unique(edge_index[1])
        self.num_users = len(self.user_ids)
        self.num_items = len(self.item_ids)

    def __len__(self):
        return self.edge_index.shape[-1]

    def __getitem__(self, idx):
        user_id = self.edge_index[0, idx].item()
        item_id = self.edge_index[1, idx].item()
        user_feature = self.user_features[user_id, :]
        item_feature = self.item_features[item_id, :]
        rating = self.ratings[idx].item()

        return (user_id, item_id, user_feature, item_feature, rating)


class DatasetLoader:
    def __init__(self, dataset):
        if dataset in ["MovieLens1M-ranking", "Frappe"]:
            train_file_name = f"datasets/{dataset}-processed/train.pt"
            self.train_set = torch.load(train_file_name)

            val_file_name = f"datasets/{dataset}-processed/val.pt"
            self.val_set = torch.load(val_file_name)

            test_file_name = f"datasets/{dataset}-processed/test.pt"
            self.test_set = torch.load(test_file_name)

        else:
            train_file_name = f"datasets/{dataset}-processed/train.pt"
            self.train_set = UserItemRatingDataset(torch.load(train_file_name))

            val_file_name = f"datasets/{dataset}-processed/val.pt"
            self.val_set = UserItemRatingDataset(torch.load(val_file_name))

            test_file_name = f"datasets/{dataset}-processed/test.pt"
            self.test_set = UserItemRatingDataset(torch.load(test_file_name))

    def get_datasets(self):
        return self.train_set, self.val_set, self.test_set


class MovieLens20MDataset(Dataset):
    def __init__(self, h5_file_path, dataset_key):
        self.h5_file_path = h5_file_path
        self.dataset_key = dataset_key
        self.file = None  # File handle to be opened lazily

        # Open the file temporarily to get the dataset length
        with h5py.File(self.h5_file_path, "r") as f:
            self.data_len = f[f"{self.dataset_key}/block0_values"].shape[0]

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        # Open the HDF5 file lazily (if not already opened)
        if self.file is None:
            self.file = h5py.File(self.h5_file_path, "r")

        # Access the data at the given index
        user_id = int(self.file[f"{self.dataset_key}/block0_values"][index, 0])
        item_id = int(self.file[f"{self.dataset_key}/block0_values"][index, 1])
        rating = self.file[f"{self.dataset_key}/block0_values"][index, 2]
        dense_features = self.file[f"{self.dataset_key}/block0_values"][index, 3:23]
        sparse_features = self.file[f"{self.dataset_key}/block0_values"][index, 23:38]

        return (
            user_id,
            item_id,
            torch.tensor(dense_features, dtype=torch.float32),
            torch.tensor(sparse_features, dtype=torch.int64),
            rating,
        )

    def __del__(self):
        # Ensure the file is closed when the dataset object is deleted
        if self.file is not None:
            self.file.close()


class FrappeDataset(Dataset):
    def __init__(self, user_list, item_list, sparse_features, ratings):
        self.user_list = user_list
        self.item_list = item_list
        self.sparse_features = sparse_features
        self.ratings = ratings

    def __len__(self):
        return self.user_list.shape[-1]

    def __getitem__(self, idx):
        user_id = self.user_list[idx].item()
        item_id = self.item_list[idx].item()
        sparse_features = self.sparse_features[idx]
        rating = self.ratings[idx].item()

        return user_id, item_id, sparse_features, rating
