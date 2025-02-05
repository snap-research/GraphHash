import numpy as np


def first_index_in_list1(list1, list2):
    for index, element in enumerate(list1):
        if element in list2:
            return index
    return None  # Return None if no element from list1 is found in list2


def indices_in_list1(list1, list2, k):
    res = np.zeros(k)
    for index, element in enumerate(list1):
        if element in list2:
            res[index] = 1
    return res


def IDCG(real_k, k):
    ideal_ranking = np.zeros(k)
    ideal_ranking[:real_k] = 1
    return (ideal_ranking / np.log2(np.arange(1, k + 1) + 1)).sum()


def quality_metrics(topk_predictions, test_candidates, k):
    test_users = list(test_candidates.keys())

    user_metrics = np.zeros((len(test_users), 2))

    for i, user in enumerate(test_users):
        correct = sum(
            (1 for item in topk_predictions[i] if item in test_candidates[user])
        )

        # Recall

        real_k = min(k, len(test_candidates[user]))

        user_recall = correct / real_k

        # NDCG

        all_relevant = indices_in_list1(topk_predictions[i], test_candidates[user], k)

        user_dcg = np.sum(all_relevant / np.log2(np.arange(2, k + 2)))

        user_idcg = IDCG(real_k, k)

        user_ndcg = user_dcg / user_idcg

        user_metrics[i, :] = [user_recall, user_ndcg]

    return user_metrics
