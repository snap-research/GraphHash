import os
import torch
import numpy as np
from utils import save_checkpoint
from tqdm import tqdm
import wandb
from torcheval.metrics import BinaryAUROC


def train_model(
    dataset,
    model,
    train_loader,
    user_cluster_tensors,
    item_cluster_tensors,
    criterion,
    optimizer,
    device,
):
    model.train()
    running_loss = 0.0
    total_num_users = 0

    for inputs in tqdm(train_loader):
        if dataset not in ["Frappe"]:
            user_id, item_id, user_feature, item_feature, rating = inputs

            user_cluster_ids = user_cluster_tensors[user_id]
            item_cluster_ids = item_cluster_tensors[item_id]

            user_feature = user_feature.to(device)
            item_feature = item_feature.to(device)
            rating = rating.to(device)

            optimizer.zero_grad()

            output = model(
                (user_cluster_ids, item_cluster_ids, user_feature, item_feature)
            ).flatten()

        elif dataset == "Frappe":
            user_id, item_id, sparse_features, rating = inputs

            user_cluster_ids = user_cluster_tensors[user_id]
            item_cluster_ids = item_cluster_tensors[item_id]

            sparse_features = sparse_features.to(device)
            rating = rating.to(device)

            optimizer.zero_grad()

            output = model(
                (user_cluster_ids, item_cluster_ids, sparse_features)
            ).flatten()

        loss = criterion(output, rating.to(torch.float32))

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * len(user_cluster_ids)
        total_num_users += len(user_cluster_ids)

    # Calculate epoch loss and accuracy
    epoch_loss = running_loss / total_num_users

    return epoch_loss


def evaluate_model(
    dataset,
    model,
    val_loader,
    user_cluster_tensors,
    item_cluster_tensors,
    criterion,
    device,
    prediction=False,
):
    model.eval()
    running_loss = 0.0
    total_num_users = 0

    auc = BinaryAUROC()
    output_rec = []

    with torch.no_grad():
        for inputs in val_loader:
            if dataset not in ["Frappe"]:
                user_id, item_id, user_feature, item_feature, rating = inputs

                user_feature = user_feature.to(device)
                item_feature = item_feature.to(device)
                rating = rating.to(device)

                # Retrieve precomputed cluster tensors
                user_cluster_ids = user_cluster_tensors[user_id]
                item_cluster_ids = item_cluster_tensors[item_id]

                if dataset == "MovieLens20M":
                    output = model(
                        (user_cluster_ids, item_cluster_ids, item_feature, user_feature)
                    ).flatten()
                else:
                    output = model(
                        (
                            user_cluster_ids,
                            item_cluster_ids,
                            user_feature.to(torch.float32),
                            item_feature,
                        )
                    ).flatten()

            elif dataset == "Frappe":
                user_id, item_id, sparse_features, rating = inputs

                user_cluster_ids = user_cluster_tensors[user_id]
                item_cluster_ids = item_cluster_tensors[item_id]

                sparse_features = sparse_features.to(device)
                rating = rating.to(device)

                output = model(
                    (user_cluster_ids, item_cluster_ids, sparse_features)
                ).flatten()

            loss = criterion(output, rating.to(torch.float32))

            auc.update(output, rating)

            if prediction:
                output_rec.append(output)

            running_loss += loss.item() * len(user_cluster_ids)
            total_num_users += len(user_cluster_ids)

        # Calculate epoch loss and accuracy
        epoch_loss = running_loss / total_num_users

        auc_value = auc.compute()

        if prediction:
            output_rec = torch.cat(output_rec)

            return epoch_loss, auc_value, output_rec

        else:
            return epoch_loss, auc_value


def train_and_evaluate_ranking(
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
    num_epochs=20,
    patience=5,
):
    best_val_auc = 0
    patience_counter = 0

    user_cluster_tensors = torch.tensor(user_clusters, dtype=torch.int64, device=device)
    item_cluster_tensors = torch.tensor(item_clusters, dtype=torch.int64, device=device)

    for epoch in tqdm(range(num_epochs)):
        train_loss = train_model(
            dataset,
            model,
            train_loader,
            user_cluster_tensors,
            item_cluster_tensors,
            criterion,
            optimizer,
            device,
        )
        val_loss, val_auc_value = evaluate_model(
            dataset,
            model,
            val_loader,
            user_cluster_tensors,
            item_cluster_tensors,
            criterion,
            device,
        )

        print(f"Train Loss:{train_loss}, Val Loss:{val_loss}, Val AUC:{val_auc_value}")
        wandb.log(
            {
                "Train Loss": train_loss,
                "Val Loss": val_loss,
                "Val AUC": val_auc_value,
            }
        )

        # Early stopping
        if patience_counter >= patience:
            break

        if val_auc_value > best_val_auc:
            best_val_auc = val_auc_value
            patience_counter = 0
            save_checkpoint(model, run_name)

        else:
            patience_counter += 1

    # Evaluate on test set
    checkpoint = torch.load(f"results/{run_name}/best_model.pth")
    model.load_state_dict(checkpoint)

    test_loss, test_auc_value, test_pred = evaluate_model(
        dataset,
        model,
        test_loader,
        user_cluster_tensors,
        item_cluster_tensors,
        criterion,
        device,
        prediction=True,
    )

    print(f"Test Loss:{test_loss}, Test AUC:{test_auc_value}")

    wandb.log(
        {
            "Test Loss": test_loss,
            "Test AUC": test_auc_value,
        }
    )
