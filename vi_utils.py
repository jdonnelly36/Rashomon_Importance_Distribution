import numpy as np
import time
import copy
from collections import deque
from sklearn.linear_model import LinearRegression as LinearRegression
import sys
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
    eps=1e-6, num_perts=10,
    for_joint=False
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

    original_preds = np.array(trie_predict_recursive(x_all, trie, np.zeros(x_all.shape[0])))
    # map from -2, -1 version to 0, 1 labels
    original_preds = -1 * (original_preds + 1)
    original_acc = (np.repeat(y_all.reshape((1, -1)), original_preds.shape[0], axis=0) == original_preds).mean(axis=1)
    
    perturbed_acc = 0
    for _ in range(num_perts):
        x_perturbed, y_perturbed = perturb_divided(data_df.to_numpy(), var_of_interest)
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
