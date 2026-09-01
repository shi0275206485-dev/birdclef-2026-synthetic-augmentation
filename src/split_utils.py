"""
Compliant train/validation split (shared by the three models in Group A, ensuring comparability).

Assignment requirement: classes with only 1 sample must always go into the validation set.
This function's logic:
  1. Count the number of samples per class;
  2. Classes with too few samples to allow a group split (default <2, i.e. single-sample classes)
     have all of their samples forced into the validation set;
  3. The remaining classes are split via GroupShuffleSplit(80/20) grouped by filename, to prevent
     the same recording from appearing in both train and val (leakage prevention).

Returns train_idx, val_idx, which can be used directly for X[train_idx], etc.
"""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def compliant_split(y, groups, test_size=0.2, random_state=42,
                    min_train_count=2, verbose=True):
    """
    y       : 1D array, integer labels
    groups  : 1D array/Series, the recording/file each sample belongs to (leakage-prevention grouping key)
    min_train_count : when a class's sample count is < this value, the whole class is pushed into the
                       validation set (default 2, i.e. single-sample classes)
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    n = len(y)
    all_idx = np.arange(n)

    # ---- 1. Find "too few samples" classes (single-sample classes, etc.) ----
    classes, counts = np.unique(y, return_counts=True)
    rare_classes = classes[counts < min_train_count]
    rare_mask = np.isin(y, rare_classes)

    forced_val_idx = all_idx[rare_mask]      # these are forced into the validation set
    remaining_idx = all_idx[~rare_mask]      # the rest take part in the normal split

    if verbose:
        print(f"[split] Total samples: {n}, total classes: {len(classes)}")
        print(f"[split] Classes with sample count < {min_train_count}: {len(rare_classes)} "
              f"-> these classes ({len(forced_val_idx)} samples) are forced into the validation set")

    # ---- 2. Split the remaining samples 80/20 grouped by filename ----
    if len(remaining_idx) == 0:
        raise ValueError("No samples available for training, check the data.")

    gss = GroupShuffleSplit(test_size=test_size, n_splits=1,
                            random_state=random_state)
    sub_train, sub_val = next(
        gss.split(remaining_idx, y[remaining_idx], groups[remaining_idx])
    )
    train_idx = remaining_idx[sub_train]
    val_idx = remaining_idx[sub_val]

    # ---- 3. Merge the forced-validation samples into val ----
    val_idx = np.concatenate([val_idx, forced_val_idx])

    if verbose:
        print(f"[split] Train: {len(train_idx)}  Val: {len(val_idx)} "
              f"(of which {len(forced_val_idx)} are rare-class samples forced into val)")
        # Check: does the training set cover most of the classes?
        n_train_cls = len(np.unique(y[train_idx]))
        print(f"[split] Classes covered by training set: {n_train_cls}/{len(classes)} "
              f"(the {len(classes)-n_train_cls} missing classes are only in the validation set, "
              f"Group A is expected to misclassify them)")

    return train_idx, val_idx