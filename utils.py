import torch
from scipy.sparse import csr_matrix
import numpy as np
from sknetwork.clustering import Louvain
import math
from lshashpy3 import LSHash
import os
import random
from sympy import nextprime



def training_graph(train_set, num_total_users, num_total_items):
    users_list = []
    items_list = []

    for user_id, item_id, *rest in train_set:
        users_list.append(user_id)
        items_list.append(item_id)

    entries = np.ones(len(users_list))

    biadjacency = csr_matrix(
        (entries, (users_list, items_list)), shape=(num_total_users, num_total_items)
    )

    return biadjacency


def graphClustering(biadjacency, resolution):
    louvain = Louvain(
        resolution=resolution
    )  # we could try different value like 1, 3, 10
    louvain.fit(biadjacency, force_bipartite=True)
    users_cluster = louvain.labels_row_
    items_cluster = louvain.labels_col_
    return (
        users_cluster,
        items_cluster,
        len(set(users_cluster)),
        len(set(items_cluster)),
    )



def map_to_consecutive_integers(values):
    """
    Maps each unique value in the input list to a unique consecutive integer.

    Args:
    values (list): The list of values to be mapped.

    Returns:
    tuple: A tuple containing the mapped list and the dictionary of value to integer mapping.
    """
    # Get unique values and sort them
    unique_values = sorted(set(values))

    # Create a mapping from unique values to consecutive integers
    value_to_int = {val: idx for idx, val in enumerate(unique_values)}

    # Map original values to the consecutive integers
    mapped_values = [value_to_int[val] for val in values]

    return np.array(mapped_values)


def candidate_set(dataset):
    user_to_relevant_items = {}
    for user_id, item_id, *rest in dataset:
        if user_id not in user_to_relevant_items:
            user_to_relevant_items[user_id] = []
        user_to_relevant_items[user_id].append(item_id)
    return user_to_relevant_items

def candidate_ground_truth(dataset):
    user_to_relevant_items_ratings = {}
    for user_id, item_id, _, _, rating in dataset:
        if user_id not in user_to_relevant_items_ratings:
            user_to_relevant_items_ratings[user_id] = []
        user_to_relevant_items_ratings[user_id].append(rating)
    return user_to_relevant_items_ratings


def merge_dicts(dict1, dict2):
    # Initialize the result dictionary
    merged_dict = {}

    # Merge dict1 into the result dictionary
    for key, value in dict1.items():
        if key not in merged_dict:
            merged_dict[key] = value
        else:
            merged_dict[key].extend(value)

    # Merge dict2 into the result dictionary
    for key, value in dict2.items():
        if key not in merged_dict:
            merged_dict[key] = value
        else:
            merged_dict[key].extend(value)

    return merged_dict


def generate_user_mask(k, indices_to_zero):
    # Initialize a vector of length k with all ones
    vector = torch.zeros(k)

    # Set the specified indices to zero
    vector[indices_to_zero] = -5e10

    return vector.reshape(1, -1)




def negative_candidates(train_candidates, num_total_users, num_total_items):
    user_to_irrelevant_items = {}

    for user in range(num_total_users):
        if user not in train_candidates:
            user_to_irrelevant_items[user] = np.arange(num_total_items)
        else:
            user_to_irrelevant_items[user] = [
                item
                for item in np.arange(num_total_items)
                if item not in train_candidates[user]
            ]

    return user_to_irrelevant_items




def universal_hash(key, key_max, m, bit_length=60):

    # Calculate the maximum value for the specified bit length
    max_value = (1 << bit_length) - 1  # This is equivalent to 2^bit_length - 1

    # Generate a random number in the range [m+1, max_value]
    random_number = random.randint(key_max, max_value)
    
    # Ensure the number is odd (since even numbers greater than 2 are not prime)
    random_number |= 1
    
    # Find the next prime greater than or equal to the random number
    prime_candidate = nextprime(random_number)



    p = prime_candidate  # A large prime number
    a = np.random.randint(1, p, dtype=np.int64)  # Random integer between 1 and p-1
    b = np.random.randint(0, p, dtype=np.int64)  # Random integer between 0 and p-1

    hash_value = ((a * key + b) % p) % m

    return hash_value


def double_frequency_hash_code(n, fixed_indices, hash_size):
    # Create the array from 1 to n
    array = np.arange(n)

    # create two hash functions:
    hash_code_1 = universal_hash(array, key_max=n, m=hash_size - len(fixed_indices))
    hash_code_2 = universal_hash(array, key_max=n, m=hash_size - len(fixed_indices))


    # Reindex the fixed set of indices from h, h+1, ..., up to the length of the set
    for i, fixed_index in enumerate(fixed_indices):
        hash_code_1[fixed_index] = (hash_size - len(fixed_indices)) + i
        hash_code_2[fixed_index] = (hash_size - len(fixed_indices)) + i

    return np.concatenate((hash_code_1.reshape(-1, 1), hash_code_2.reshape(-1, 1)), axis=1)


def frequency_hash(
    biadjacency,
    num_users_clusters,
    num_items_clusters,
    num_total_users,
    num_total_items,
    cutoff=0.5,
):
    user_frequencies = np.array(biadjacency.sum(axis=1).reshape(1, -1))[0]
    item_frequencies = np.array(biadjacency.sum(axis=0))[0]

    # Define k
    user_k = math.floor(cutoff * num_users_clusters)
    item_k = math.floor(cutoff * num_items_clusters)

    # Get the indices that would sort the array
    top_k_users = np.argsort(user_frequencies)[::-1]
    top_k_users = top_k_users[:user_k]
    top_k_items = np.argsort(item_frequencies)[::-1]
    top_k_items = top_k_items[:item_k]

    user_clusters = frequency_hash_code(
        num_total_users, top_k_users, num_users_clusters
    )
    item_clusters = frequency_hash_code(
        num_total_items, top_k_items, num_items_clusters
    )

    return user_clusters, item_clusters


def frequency_hash_code(n, top_k_ids, hash_size):
    # Create the array from 1 to n
    cluster_assignment = np.arange(n)

    # Convert top_k_ids to a set for quick lookup
    top_k_ids_set = set(top_k_ids)

    # Apply modulo operation to the rest of the indices
    for i in range(n):
        if i not in top_k_ids_set:
            cluster_assignment[i] = cluster_assignment[i] % (hash_size - len(top_k_ids))

    # Reindex the fixed set of indices from h, h+1, ..., up to the length of the set
    for i, fixed_index in enumerate(top_k_ids):
        cluster_assignment[fixed_index] = (hash_size - len(top_k_ids)) + i

    return cluster_assignment


def double_hash(num_users_clusters, num_items_clusters, num_total_users, num_total_items):
    np.random.seed(0)
    user_hash = [universal_hash(np.arange(num_total_users), key_max=num_total_users, m=num_users_clusters).reshape(-1,1) for _ in range(2)]
    item_hash = [universal_hash(np.arange(num_total_items), key_max=num_total_items, m=num_items_clusters).reshape(-1,1) for _ in range(2)]
    user_clusters = np.concatenate(user_hash, axis=1)
    item_clusters = np.concatenate(item_hash, axis=1)

    return user_clusters, item_clusters


def double_frequency_hash(
    biadjacency,
    num_users_clusters,
    num_items_clusters,
    num_total_users,
    num_total_items,
    cutoff=0.5,
):
    user_frequencies = np.array(biadjacency.sum(axis=1).reshape(1, -1))[0]
    item_frequencies = np.array(biadjacency.sum(axis=0))[0]

    # Define k
    user_k = math.floor(cutoff * num_users_clusters)
    item_k = math.floor(cutoff * num_items_clusters)

    # Get the indices that would sort the array
    top_k_users = np.argsort(user_frequencies)[::-1]
    top_k_users = top_k_users[:user_k]
    top_k_items = np.argsort(item_frequencies)[::-1]
    top_k_items = top_k_items[:item_k]

    user_clusters = double_frequency_hash_code(
        num_total_users, top_k_users, num_users_clusters
    )
    item_clusters = double_frequency_hash_code(
        num_total_items, top_k_items, num_items_clusters
    )

    return user_clusters, item_clusters




def lsh_get_hashes(lsh, input_point):
        """ Takes a single input point `input_point`, iterate through the
        uniform planes, and returns a list with size of `num_hashtables`
        containing the corresponding hash for each hashtable.

        :param input_point:
            A list, or tuple, or numpy ndarray object that contains numbers
            only. The dimension needs to be 1 * `input_dim`.
        """
        
        hashes = []
        for i, table in enumerate(lsh.hash_tables):
            binary_hash = lsh._hash(lsh.uniform_planes[i], input_point)
            hashes.append(binary_hash)
        
        return hashes



# this would give more buckets than num_users_clusters, num_items_clusters
def lsh(user_features,
        item_features,
        num_users_clusters, 
        num_items_clusters):
    
    user_features = user_features.numpy()
    item_features = item_features.numpy()
    
    user_hash_size = len(bin(num_users_clusters)[2:])
    item_hash_size = len(bin(num_items_clusters)[2:])

    user_lsh = LSHash(hash_size=user_hash_size, input_dim=user_features.shape[-1], num_hashtables=1)
    item_lsh = LSHash(hash_size=item_hash_size, input_dim=item_features.shape[-1], num_hashtables=1)

    user_clusters = []
    item_clusters = []
    for i in range(user_features.shape[0]):
        user_id = int(lsh_get_hashes(user_lsh, user_features[i,:])[0], 2)
        user_clusters.append(user_id)

    for j in range(item_features.shape[0]):
        item_id = int(lsh_get_hashes(item_lsh, item_features[j,:])[0], 2)
        item_clusters.append(item_id)

    return np.array(user_clusters), np.array(item_clusters), np.max(user_clusters)+1, np.max(item_clusters)+1



def save_checkpoint(model, run_name):
    # Define the directory and file path
    dir_path = f"results/{run_name}"

    # Check if the directory exists, if not, create it
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    torch.save(model.state_dict(), f'results/{run_name}/best_model.pth')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def check_gradient_flow(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.abs().mean()
            grad_max = param.grad.abs().max()
            print(f"Layer: {name} | Grad mean: {grad_mean} | Grad max: {grad_max}")
        else:
            print(f"Layer: {name} | No gradient found.")

def double_graph_hash(biadjacency, resolution, num_users_clusters, num_items_clusters, num_total_users, num_total_items):
    np.random.seed(0)
    (
        user_clusters,
        item_clusters,
        num_users_clusters,
        num_items_clusters,
    ) = graphClustering(biadjacency=biadjacency, resolution=resolution)
    user_clusters = map_to_consecutive_integers(user_clusters)
    item_clusters = map_to_consecutive_integers(item_clusters)
    user_hash = [universal_hash(np.arange(num_total_users), key_max=num_total_users, m=num_users_clusters).reshape(-1,1), user_clusters.reshape(-1,1)]
    item_hash = [universal_hash(np.arange(num_total_items), key_max=num_total_items, m=num_items_clusters).reshape(-1,1), item_clusters.reshape(-1,1)]
    user_clusters = np.concatenate(user_hash, axis=1)
    item_clusters = np.concatenate(item_hash, axis=1)
    return user_clusters, item_clusters




