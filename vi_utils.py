import numpy as np
import time
import copy
from collections import deque
from sklearn.linear_model import LinearRegression as LinearRegression
import sys
import hashlib
from typing import Dict, List
sys.path.append('./')
from rashomon_importance_utils import trie_predict_recursive

def perturb_divided(data, target_col):
    """Perturb column target_col of data using e_divide strategy

    Args:
        data (array): Data set
        target_col (int or list of ints): Index or indices of the column(s) to mess with
    Returns:
        data_perturbed: A copy of X with column target_col perturbed
    """
    n_samples = data.shape[0]

    # Create a copy of our given data
    data_copy = copy.deepcopy(data)
    # Shuffle along the first dimension to make sure
    # e_divide will be valid
    np.random.shuffle(data_copy)

    # Grab our X and Y components
    x_copy = data_copy[:, :-1]
    x_copy_src = copy.deepcopy(x_copy)
    y_copy = data_copy[:, -1]

    if n_samples % 2 == 0:
        x_copy[:n_samples // 2, target_col] = x_copy_src[n_samples // 2:, target_col]
        x_copy[n_samples // 2:, target_col] = x_copy_src[:n_samples // 2, target_col]
    else:
        x_copy[n_samples // 2:, target_col] = x_copy_src[:n_samples // 2+1, target_col]
        x_copy[:n_samples // 2, target_col] = x_copy_src[n_samples // 2+1:, target_col]


    return x_copy, y_copy

def get_model_reliances(
    trie, data_df, var_of_interest=0, 
    eps=1e-6, num_perts=1,
    for_joint=False,
    cached_row_predictions=None
):
    """Computes the sub and div model reliance for each tree in the given trie
    over the given dataset

    Args:
        trie (str): Rashomon trie json
        data_df (pd.DataFrame): Dataframe of the dataset to compute tree accuracies
        var_of_interest (int or [int]): The column or columns to compute
            model relianc for
        weight_by_eps (bool): 
        num_perts (int): The number of permutations to consier when computing
            model reliance
        feature_description (dict): A dictionary that maps feature name to their
            descriptions. If it is not given, original feature names will be
            used (might be hard for readers to understand).
        cached_row_predictions (PredictionCache, optional): A cache that maps from the
            hash of a X value to the vector returned by trie_predict.
            If None, don't do anything with this

    Returns:
        div_model_reliances: Dictionary of div model reliance values, of the form
        {
            means: [mean],
            observed_mr_1: [proportion of R set realizing observed_mr_1],
            observed_mr_2: [proportion of R set realizing observed_mr_2],
            ...
        }
        sub_model_reliances: Dictionary of sub model reliance values, of the form
        {
            means: [mean],
            observed_mr_1: [proportion of R set realizing observed_mr_1],
            observed_mr_2: [proportion of R set realizing observed_mr_2],
            ...
        }
        num_models: The number of models in this trie
    """
    
    # Count samples and accuracies of trees in place
    # Extract the x data from the dataframe (here we use all data to evaluate)
    x_all = data_df.to_numpy()[:, 0 : data_df.shape[1] - 1]
    y_all = data_df.to_numpy()[:, data_df.shape[1] - 1]

    if cached_row_predictions is not None:
        original_preds = predict_with_cache(
            x_all,
            cached_row_predictions,
            lambda x_cur: np.stack(trie_predict_recursive(x_cur, trie, np.zeros(x_cur.shape[0])), axis=-1)
        ).transpose()
    else:
        original_preds = np.array(trie_predict_recursive(x_all, trie, np.zeros(x_all.shape[0])))
    # map from -2, -1 version to 0, 1 labels
    original_preds = -1 * (original_preds + 1)
    original_acc = (np.repeat(y_all.reshape((1, -1)), original_preds.shape[0], axis=0) == original_preds).mean(axis=1)
    
    perturbed_acc = 0
    for _ in range(num_perts):
        x_perturbed, y_perturbed = perturb_divided(data_df.to_numpy(), var_of_interest)

        if cached_row_predictions is not None:
            pert_preds = predict_with_cache(
                x_perturbed,
                cached_row_predictions,
                lambda x_cur: np.stack(trie_predict_recursive(x_cur, trie, np.zeros(x_cur.shape[0])), axis=-1)
            ).transpose()
        else:
            pert_preds = np.array(trie_predict_recursive(x_perturbed, trie, np.zeros(x_perturbed.shape[0])))
        # map from -2, -1 version to 0, 1 labels
        pert_preds = -1 * (pert_preds + 1)
        perturbed_acc += (np.repeat(y_perturbed.reshape((1, -1)), pert_preds.shape[0], axis=0) == pert_preds).mean(axis=1)

    perturbed_acc /= num_perts
    
    # Misclassification loss is just 1 - accuracy
    div_mrs = (1-perturbed_acc+eps) / (1-original_acc+eps)
    sub_mrs = (1-perturbed_acc) - (1-original_acc)

    div_model_reliances = {}
    sub_model_reliances = {}
    #objective_lists = {}

    num_models = div_mrs.shape[0]
    # Loss is something like cur_tree[-2]
    #mr_values = np.array(p_map(par_friendly_iter, tids, num_cpus=num_cpus))
    #print(f"Got through par step in {time.time() - start} seconds")
    # Once converted to an array, mr_values[:, 0] is all the div_mrs,
    # mr_values[:, 1] is all the subs, and  mr_values[:, 2] is all the accs
    #print("Ran acc stuff in {} seconds".format(time.time() - start))
    if for_joint:
        return div_mrs, sub_mrs, num_models

    start = time.time()
    '''
    For each unique div mr
    After this loop, div_model_reliances is a dictionary of the form
    {
        means: [mean],
        observed_mr_1: [proportion of R set realizing observed_mr_1],
        observed_mr_2: [proportion of R set realizing observed_mr_2],
        ...
    }
    '''
    running_mean = 0
    for val in np.unique(div_mrs):
        div_model_reliances[val] = div_mrs[div_mrs == val].shape[-1] / num_models
        running_mean += div_model_reliances[val] * val
    div_model_reliances['means'] = [running_mean]

    running_mean = 0
    for val in np.unique(sub_mrs):
        sub_model_reliances[val] = sub_mrs[sub_mrs == val].shape[-1] / num_models
        running_mean += sub_model_reliances[val] * val
    sub_model_reliances['means'] = [running_mean]
    
    #print("Ran first processing loop in {} seconds".format(time.time() - start))
    return div_model_reliances, sub_model_reliances, num_models#, objective_lists

def hash_rows(X: np.ndarray) -> np.ndarray:
    """
    Compute a stable hash for each row of X.
    Returns an array of strings of length n_samples.
    """
    if not X.flags['C_CONTIGUOUS']:
        X = np.ascontiguousarray(X)

    hashes = []
    for row in X:
        h = hashlib.sha1(row.tobytes()).hexdigest()
        hashes.append(h)
    return np.array(hashes)

class PredictionCache:
    def __init__(self, n_trees):
        self.n_trees = n_trees
        self.cache: Dict[str, np.ndarray] = {}

    def has_prediction(self, row_hash: str) -> bool:
        return row_hash in self.cache

    def get_predictions(self, row_hash: str):
        return self.cache[row_hash]

    def set_prediction(self, row_hash: str, values):
        if row_hash not in self.cache:
            self.cache[row_hash] = values

def predict_with_cache(
    X: np.ndarray,
    cache: PredictionCache,
    prediction_fn,
) -> np.ndarray:
    """
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    cache : PredictionCache
    prediction_fn : function -- a function mapping from
        a numpy array to an array of predictions

    Returns
    -------
    preds : np.ndarray, shape (n_samples, n_models)
    """
    n_samples = X.shape[0]
    row_hashes = hash_rows(X)
    n_models = cache.n_trees

    # Output matrix
    preds = np.empty((n_samples, n_models), dtype=float)

    missing_idx = [
        i for i, h in enumerate(row_hashes)
        if not cache.has_prediction(h)
    ]

    # Predict only missing rows
    if missing_idx:
        X_missing = X[missing_idx]
        y_missing = prediction_fn(X_missing)
        for i, y in zip(missing_idx, y_missing):
            cache.set_prediction(row_hashes[i], y)

    # Fill output matrix
    for i, h in enumerate(row_hashes):
        preds[i] = cache.get_predictions(h)

    return preds
