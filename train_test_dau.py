import torch
import numpy as np
from utils import merge_dicts, save_checkpoint
from tqdm import tqdm
import wandb
import os
from metrics import quality_metrics


def train_model(
    hash_type,
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
        user_id, item_id = inputs

        user_cluster_ids = user_cluster_tensors[user_id]
        item_cluster_ids = item_cluster_tensors[item_id]

        optimizer.zero_grad()

        user_embedding, positive_item_embedding = model(
            user_cluster_ids,
            item_cluster_ids,
        )

        loss = criterion(user_embedding, positive_item_embedding)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(user_cluster_ids)
        total_num_users += len(user_cluster_ids)

    # Calculate epoch loss and accuracy
    epoch_loss = running_loss / total_num_users

    return epoch_loss


def evaluate_model(
    hash_type,
    model,
    val_loader,
    user_cluster_tensors,
    item_cluster_tensors,
    criterion,
    device,
):
    model.eval()
    running_loss = 0.0
    total_num_users = 0

    with torch.no_grad():
        for inputs in val_loader:
            user_id, item_id = inputs

            # get hashed id
            user_cluster_ids = user_cluster_tensors[user_id]
            item_cluster_ids = item_cluster_tensors[item_id]

            user_embedding, positive_item_embedding = model(
                user_cluster_ids,
                item_cluster_ids,
            )

            loss = criterion(user_embedding, positive_item_embedding)
            running_loss += loss.item() * len(user_cluster_ids)
            total_num_users += len(user_cluster_ids)

    # Calculate epoch loss and accuracy
    epoch_loss = running_loss / total_num_users

    return epoch_loss


def train_and_evaluate_dau(
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
    num_epochs=20,
    patience=5,
):
    best_val_recall = 0
    patience_counter = 0

    if type(model).__name__ == "LightGCN":
        if len(user_clusters.shape) == 1:
            user_cluster_tensors = torch.arange(
                len(user_clusters), dtype=torch.int64, device=device
            )
            item_cluster_tensors = torch.arange(
                len(item_clusters), dtype=torch.int64, device=device
            )
        else:
            user_cluster_tensors = torch.arange(
                user_clusters.shape[0], dtype=torch.int64, device=device
            )
            item_cluster_tensors = torch.arange(
                item_clusters.shape[0], dtype=torch.int64, device=device
            )

    else:
        user_cluster_tensors = torch.tensor(
            user_clusters, dtype=torch.int64, device=device
        )
        item_cluster_tensors = torch.tensor(
            item_clusters, dtype=torch.int64, device=device
        )

    for epoch in tqdm(range(num_epochs)):
        train_loss = train_model(
            hash_type,
            model,
            train_loader,
            user_cluster_tensors,
            item_cluster_tensors,
            criterion,
            optimizer,
            device,
        )

        if epoch % 10 == 0:
            with torch.no_grad():
                masked_topk_predictions = model.recommend(
                    hash_type=hash_type,
                    train_candidates=train_candidates,
                    test_candidates=val_candidates,
                    user_clusters=user_cluster_tensors,
                    item_clusters=item_cluster_tensors,
                    device=device,
                    batch_size=20480,
                    k=k,
                )

            user_metrics = quality_metrics(masked_topk_predictions, val_candidates, k)

            val_recall = user_metrics.mean(axis=0)[0]
            val_ndcg = user_metrics.mean(axis=0)[1]

            print(
                f"Train Loss:{train_loss}, Val Recall: {val_recall}, Val NDCG: {val_ndcg}"
            )
            wandb.log(
                {
                    "Train Loss": train_loss,
                    "Val Recall": val_recall,
                    "Val NDCG": val_ndcg,
                }
            )

        else:
            print(f"Train Loss:{train_loss}")

            wandb.log(
                {
                    "Train Loss": train_loss,
                }
            )

        # Early stopping
        if patience_counter >= patience:
            break

        if val_recall > best_val_recall:
            best_val_recall = val_recall
            patience_counter = 0
            save_checkpoint(model, run_name)

        else:
            patience_counter += 1

    # Evaluate on test set
    checkpoint = torch.load(f"results/{run_name}/best_model.pth")
    model.load_state_dict(checkpoint)

    train_val_candidates = merge_dicts(train_candidates, val_candidates)

    with torch.no_grad():
        masked_topk_predictions = model.recommend(
            hash_type=hash_type,
            train_candidates=train_val_candidates,
            test_candidates=test_candidates,
            user_clusters=user_cluster_tensors,
            item_clusters=item_cluster_tensors,
            device=device,
            batch_size=20480,
            k=k,
        )

    user_metrics = quality_metrics(masked_topk_predictions, test_candidates, k)

    test_recall = user_metrics.mean(axis=0)[0]
    test_ndcg = user_metrics.mean(axis=0)[1]

    wandb.log({"Test Recall": test_recall, "Test NDCG": test_ndcg})

    # save the predictions and user_metrics for later analysis: uncomment if needed

    # # Define the directory and file path
    # dir_path = f"results/{run_name}"

    # # Check if the directory exists, if not, create it
    # if not os.path.exists(dir_path):
    #     os.makedirs(dir_path)

    # torch.save(masked_topk_predictions, f"results/{run_name}/test_recommendations.pt")
    # np.save(f"results/{run_name}/train_candidates.npy", train_candidates)
    # np.save(f"results/{run_name}/test_candidates.npy", test_candidates)
    # np.save(f"results/{run_name}/user_metrics_all.npy", user_metrics)
