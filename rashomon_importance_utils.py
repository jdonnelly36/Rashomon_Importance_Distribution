import json
import os
import numpy as np
import pandas as pd
from treefarms import TREEFARMS

def construct_tree_rset(
    dataset_path,
    lam, db, eps,
    original_config_path='./config.json',
    save_dir=None,
    config_idx=None,
    verbose=False
    ):

    df = pd.read_csv(dataset_path)
    if verbose:
        print("save_dir: {}".format(save_dir))
        print("config_idx: {}".format(config_idx))

    with open(original_config_path) as f:
        configJson = json.load(f)
        configJson['depth_budget'] = db

        configJson['rashomon_bound_multiplier'] = 0
        configJson['rashomon_bound'] = 0
        configJson['rashomon_bound_adder'] = eps

        configJson['regularization'] = lam
        
        configJson['rashomon_trie'] = os.path.join(save_dir, 
                                        f'trie_bootstrap_{config_idx}_eps_{eps}_db_{db}_reg_{lam}.json')
        configJson['rashomon_model_set_suffix'] = f'_bootstrap_{config_idx}_eps_{eps}_db_{db}_reg_{lam}.json'

        configJson['verbose'] = verbose
        
        if verbose:
            print('Running TreeFarms')
            
        tf = TREEFARMS(configJson)
        tf.fit(df.iloc[:, :-1], df.iloc[:, -1])
    f.close()

def trie_predict_recursive(dataset, trie, node_assignments):
    """Get the prediction vector for each tree in trie
    on the given dataset. -3 represents a tmp value,
    -2 represents a positive prediction, -1 negative

    Args:
        dataset (np.array): A (n_samples, n_features) binary matrix
        trie (dict): The trie of interest
        node_assignments (np.array): An array of shape (n_samples)
            indicating which split each sample goes to in this subtree

    Returns:
        list: list of prediction vectors
    """
    # Base case: We've reached a leaf
    if 'objective' in list(trie.keys()):
        return [node_assignments]
    
    # Keep a list of overall predictions
    overall_preds = []
    
    # Iterate over possible subtrees
    for key in trie:
        new_node_assignments = np.ones_like(node_assignments) * -3
        new_node_assignments[node_assignments < 0] = node_assignments[node_assignments < 0]
        
        # Grab splits in this subtree
        splits = np.array([k for k in key.split(' ')])
        
        # Keep track of how much we need to adjust for 
        # dead branches
        offset_for_leaves = 0
        for split_ind, split in enumerate(splits):
            split = int(split)
            cur_split_mask = node_assignments == split_ind
            
            if split == -1:
                new_node_assignments[cur_split_mask] = split
                offset_for_leaves += 1
            elif split == -2:
                new_node_assignments[cur_split_mask] = split
                offset_for_leaves += 1
            else:
                l_split_mask = dataset[:, split] != 0

                new_node_assignments[cur_split_mask & l_split_mask] = (split_ind - offset_for_leaves) * 2
                new_node_assignments[cur_split_mask & (~l_split_mask)] = (split_ind - offset_for_leaves) * 2 + 1
        
        assert not (-3 in new_node_assignments), new_node_assignments
        overall_preds = overall_preds + trie_predict_recursive(dataset, trie[key], new_node_assignments)
    return overall_preds
