## RID Documentation
This codebase contains the code necessary to compute the Rashomon Importance Distribution as described in "The Rashomon Importance Distribution: Getting RID of Unstable, Single Model-Based Variable Importance". 

### Environment Setup
In order to configure your environment, you will need to:
1. Install all required python packages via `pip install -r requirements.txt`

### Computing RID
A simple demonstration of the RID interface can be seen in `example.ipynb`. The primary interface for RID is the `RashomonImportanceDistribution` class, which computes RID in its constructor and provides functions to examine RID. The following parameters are available in the `RashomonImportanceDistribution`constructor:

_**input_df**_ -- A pandas DataFrame containing a **binarized** version of the dataset we seek to explain
    
_**binning_map**_ -- A dictionary of the form `{0: [0, 1, 2], 1: [3, 4], 2: [5, 6, 7, 8]}` describing which variables in the original, unbinned version of the dataset map to which columns of the binarized version `input_df`. The example given states that the 0-th variable in the original data is represented by bins 0, 1, and 2, and so on.

_**db**_ -- The maximum depth allowed for the decision trees in our rashomon sets. Note that large values can cause computational problems

_**lam**_ -- The regularization weight to use when computing rashomon sets

_**eps**_ -- The threshold to use when computing rashomon sets (i.e., models within eps of optimal are included)

_**dataset_name**_ -- The name of the datset being analyzed. Used to determine where to cache various files

_**n_resamples**_ -- The number of bootstrap samples to compute

_**cache_dir_root**_ -- The root file path at which all cached files should be stored

_**rashomon_output_dir**_ -- The name of the subfolder of cache_dir_root in which rashomon sets will be stored

_**verbose**_ -- Whether to produce extra logging

_**vi_metric**_ -- The VI metric to use for this RID; should be one of ['sub_mr', 'div_mr', 'sub_cmr', 'div_cmr']
    
_**max_par_for_gosdt**_ -- The maximum number of instances of GOSDT to run in parallell; reduce this number if memory issues occur
