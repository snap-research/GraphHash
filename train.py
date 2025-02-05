import wandb
import numpy as np
from scipy import sparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import (
    training_graph,
    graphClustering,
    candidate_set,
    map_to_consecutive_integers,
    frequency_hash,
    double_hash,
    double_frequency_hash,
    lsh,
    count_parameters,
    double_graph_hash,
)
from data import DatasetLoader
from train_test_id_only import train_and_evaluate
from train_test_ranking import train_and_evaluate_ranking
from train_test_dau import train_and_evaluate_dau
from train_test_ials import train_and_evaluate_ials
from loss import BPRLoss, DAULoss, iALSLoss
from data import (
    UserItemRatingDataset,
    UserItemRatingDatasetv2,
    MovieLens20MDataset,
    FrappeDataset,
)
import os
import hydra
from hydra.utils import instantiate


@hydra.main(version_base="1.2", config_path="configs", config_name="default")
def main(config):
    dataset = config.dataset.name
    embedding_dim = config.dataset.embed_dim
    num_epochs = config.epoch
    batch_size = config.dataset.bs
    learning_rate = config.dataset.lr
    weight_decay = config.dataset.wd
    patience = config.dataset.patience
    id_only = config.dataset.id_only
    k = config.dataset.k
    hash_type = config.hash_type
    dhe_k = config.dhe_k
    resolution = config.dataset.resolution
    loss = config.dataset.loss

    # Hyperparameters from command line arguments
    target = config.model._target_

    # Extract the class name by splitting the string by '.' and taking the last part
    model_name = target.split(".")[-1]

    if model_name == "LightGCN":
        num_layers = config.model.num_layers

    if loss == "DAU":
        gamma = config.dataset.gamma

    # Create a run name using the hyperparameters
    run_name = f"{model_name}_{dataset}_{loss}_hash_{hash_type}_d_{embedding_dim}_res_{resolution}_k_{k}_lr_{learning_rate}_wd_{weight_decay}_bs_{batch_size}_patience_{patience}"

    ## Initialize W&B in offline mode
    os.environ["WANDB_MODE"] = "offline"

    # Initialize W&B

    wandb.init(
        project="graph-hash",
        name=run_name,
        config={
            "model_name": model_name,
            "dataset": dataset,
            "hash_type": hash_type,
            "k": k,
            "dhe_k": dhe_k,
            "resolution": resolution,
            "lr": learning_rate,
            "wd": weight_decay,
            "loss": loss,
            "bs": batch_size,
            "patience": patience,
            "embedding_dim": embedding_dim,
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(dataset)

    # read data

    if dataset not in ["MovieLens20M"]:
        data = DatasetLoader(dataset=dataset)
        train_set, val_set, test_set = data.get_datasets()
        uid_set = set()
        iid_set = set()
        for user_id, item_id, *additional_info in train_set:
            uid_set.add(user_id)
            iid_set.add(item_id)
        num_total_users = len(uid_set)
        num_total_items = len(iid_set)
        print(num_total_users, num_total_items)

    elif dataset == "MovieLens20M":
        train_set = MovieLens20MDataset(
            h5_file_path=f"datasets/{dataset}-processed/train.h5", dataset_key="train"
        )
        val_set = MovieLens20MDataset(
            h5_file_path=f"datasets/{dataset}-processed/val.h5", dataset_key="val"
        )
        test_set = MovieLens20MDataset(
            h5_file_path=f"datasets/{dataset}-processed/test.h5", dataset_key="test"
        )

        num_total_users = config.dataset.num_total_users
        num_total_items = config.dataset.num_total_items
        print(num_total_users, num_total_items)

    if loss == "BCE":
        if dataset not in ["Frappe"]:
            user_feature_file_name = (
                f"datasets/{dataset}-processed/all_user_features.pt"
            )
            all_user_features = torch.load(user_feature_file_name)

            item_feature_file_name = (
                f"datasets/{dataset}-processed/all_item_features.pt"
            )
            all_item_features = torch.load(item_feature_file_name)

    # Infer user_feat_dim and item_feat_dim from the dataset
    if not id_only:
        if dataset not in ["Frappe"]:
            user_feature_dim = train_set[0][2].shape[-1]
            item_feature_dim = train_set[0][3].shape[-1]

    if loss in ["BPR", "DAU", "iALS"]:
        # build the relevant item set for each user
        train_candidates, val_candidates, test_candidates = (
            candidate_set(train_set),
            candidate_set(val_set),
            candidate_set(test_set),
        )

        wandb.log(
            {
                "Number of train users": len(train_candidates),
                "Number of val users": len(val_candidates),
                "Number of test users": len(test_candidates),
            }
        )

    # hashing:

    if hash_type in [
        "graph",
        "random",
        "frequency",
        "double",
        "double_frequency",
        "lsh",
        "lsh-structure",
        "spectral",
        "double_graph",
    ]:
        # Generate GH
        if dataset != "MovieLens20M":
            biadjacency = training_graph(
                train_set=train_set,
                num_total_users=num_total_users,
                num_total_items=num_total_items,
            )
        else:
            biadjacency = sparse.load_npz(
                f"datasets/{dataset}-processed/biadjacency.npz"
            )

        (
            user_clusters,
            item_clusters,
            num_users_clusters,
            num_items_clusters,
        ) = graphClustering(biadjacency=biadjacency, resolution=resolution)

        print(hash_type, num_users_clusters, num_items_clusters)

        if hash_type == "graph":
            user_clusters = map_to_consecutive_integers(user_clusters)
            item_clusters = map_to_consecutive_integers(item_clusters)

        elif hash_type == "double_graph":
            user_clusters, item_clusters = double_graph_hash(
                biadjacency,
                resolution,
                num_users_clusters,
                num_items_clusters,
                num_total_users,
                num_total_items,
            )

        elif hash_type == "random":
            user_clusters = np.arange(num_total_users) % num_users_clusters
            item_clusters = np.arange(num_total_items) % num_items_clusters

        elif hash_type == "frequency":
            user_clusters, item_clusters = frequency_hash(
                biadjacency=biadjacency,
                num_users_clusters=num_users_clusters,
                num_items_clusters=num_items_clusters,
                num_total_users=num_total_users,
                num_total_items=num_total_items,
            )

        elif hash_type == "double":
            user_clusters, item_clusters = double_hash(
                num_users_clusters=num_users_clusters,
                num_items_clusters=num_items_clusters,
                num_total_users=num_total_users,
                num_total_items=num_total_items,
            )

        elif hash_type == "double_frequency":
            user_clusters, item_clusters = double_frequency_hash(
                biadjacency=biadjacency,
                num_users_clusters=num_users_clusters,
                num_items_clusters=num_items_clusters,
                num_total_users=num_total_users,
                num_total_items=num_total_items,
            )

        elif hash_type == "lsh":
            user_clusters, item_clusters, num_users_clusters, num_items_clusters = lsh(
                user_features=all_user_features,
                item_features=all_item_features,
                num_users_clusters=num_users_clusters,
                num_items_clusters=num_items_clusters,
            )

        elif hash_type == "lsh-structure":
            biadjacency_tensor = torch.tensor(biadjacency.toarray())

            user_clusters, item_clusters, num_users_clusters, num_items_clusters = lsh(
                user_features=biadjacency_tensor,
                item_features=biadjacency_tensor.T,
                num_users_clusters=num_users_clusters,
                num_items_clusters=num_items_clusters,
            )

    elif hash_type in ["full"]:
        user_clusters = np.arange(num_total_users)
        item_clusters = np.arange(num_total_items)

        num_users_clusters, num_items_clusters = num_total_users, num_total_items

        if model_name == "LightGCN":
            biadjacency = training_graph(
                train_set=train_set,
                num_total_users=num_total_users,
                num_total_items=num_total_items,
            )

    elif hash_type == "spectral":
        user_clusters = np.load(
            f"datasets/{dataset}-processed/user_spectral_clusters.npy"
        )
        item_clusters = np.load(
            f"datasets/{dataset}-processed/item_spectral_clusters.npy"
        )

        user_clusters = map_to_consecutive_integers(user_clusters)
        item_clusters = map_to_consecutive_integers(item_clusters)

        if model_name == "LightGCN":
            biadjacency = training_graph(
                train_set=train_set,
                num_total_users=num_total_users,
                num_total_items=num_total_items,
            )

        num_total_users = np.unique(user_clusters)
        num_total_items = np.unique(item_clusters)

    wandb.log(
        {
            "Number of user clusters": num_users_clusters,
            "Number of item clusters": num_items_clusters,
        }
    )

    # Initialize model, loss function, and optimizer
    if loss in ["BPR", "DAU", "iALS"]:
        if model_name == "LightGCN":
            wandb.log({"Number of Layer": num_layers})
            model = instantiate(
                config.model,
                user_vocab_size=num_users_clusters,
                item_vocab_size=num_items_clusters,
                biadjacency=biadjacency,
                user_hashed_ids=user_clusters,
                item_hashed_ids=item_clusters,
                hash_type=hash_type,
                embedding_dim=embedding_dim,
            ).to(device)

        else:
            model = instantiate(
                config.model,
                user_vocab_size=num_users_clusters,
                item_vocab_size=num_items_clusters,
                embedding_dim=embedding_dim,
            ).to(device)

        num_parameters = count_parameters(model)
        wandb.log({"Number of model parameters": num_parameters})

        if loss == "BPR":
            criterion = BPRLoss()
        elif loss == "DAU":
            criterion = DAULoss(gamma=gamma)
        elif loss == "iALS":
            criterion = iALSLoss()
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    elif loss == "BCE":
        if dataset not in ["MovieLens20M"]:
            if dataset == "Frappe":
                model = instantiate(
                    config.model,
                    user_vocab_size=num_users_clusters,
                    item_vocab_size=num_items_clusters,
                ).to(device)
                num_parameters = count_parameters(model)
                wandb.log({"Number of model parameters": num_parameters})

            else:
                model = instantiate(
                    config.model,
                    user_vocab_size=num_users_clusters,
                    item_vocab_size=num_items_clusters,
                    embedding_dim=embedding_dim,
                    input_user_feature_size=user_feature_dim,
                    input_item_feature_size=item_feature_dim,
                ).to(device)

                num_parameters = count_parameters(model)
                wandb.log({"Number of model parameters": num_parameters})

            criterion = nn.BCELoss()
            optimizer = optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )

        elif dataset in ["MovieLens20M"]:
            model = instantiate(
                config.model,
                user_vocab_size=num_users_clusters,
                item_vocab_size=num_items_clusters,
                embedding_dim=embedding_dim,
            ).to(device)
            num_parameters = count_parameters(model)
            wandb.log({"Number of model parameters": num_parameters})

            criterion = nn.BCELoss()
            optimizer = optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )

    # Train and evaluate with early stopping

    if loss == "BPR":
        train_and_evaluate(
            hash_type,
            model,
            train_loader,
            val_loader,
            test_loader,
            user_clusters,
            item_clusters,
            train_candidates,
            val_candidates,
            test_candidates,
            k,
            criterion,
            optimizer,
            run_name,
            device,
            num_epochs=num_epochs,
            patience=patience,
        )
    elif loss == "DAU":
        train_and_evaluate_dau(
            hash_type,
            model,
            train_loader,
            val_loader,
            test_loader,
            user_clusters,
            item_clusters,
            train_candidates,
            val_candidates,
            test_candidates,
            k,
            criterion,
            optimizer,
            run_name,
            device,
            num_epochs=num_epochs,
            patience=patience,
        )
    elif loss == "iALS":
        train_and_evaluate_ials(
            hash_type,
            model,
            train_loader,
            val_loader,
            test_loader,
            user_clusters,
            item_clusters,
            train_candidates,
            val_candidates,
            test_candidates,
            k,
            criterion,
            optimizer,
            run_name,
            device,
            num_epochs=num_epochs,
            patience=patience,
        )

    elif loss == "BCE":
        train_and_evaluate_ranking(
            dataset,
            hash_type,
            model,
            train_loader,
            val_loader,
            test_loader,
            user_clusters,
            item_clusters,
            criterion,
            optimizer,
            run_name,
            device,
            num_epochs=num_epochs,
            patience=patience,
        )

    wandb.finish()


if __name__ == "__main__":
    main()
