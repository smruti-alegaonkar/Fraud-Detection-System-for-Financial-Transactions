"""
imbalance_handler.py
--------------------
Handles class imbalance using three strategies:

1. class_weight='balanced'  → Built into sklearn models (free, no resampling)
2. Random Oversampling      → Duplicates minority class examples
3. Manual SMOTE             → Synthetic Minority Oversampling Technique
                              (creates synthetic fraud examples by interpolating
                               between real fraud examples and their k-nearest
                               neighbors)
4. Random Undersampling     → Removes majority class examples

Why not just use accuracy?
   In 99.83% legit / 0.17% fraud data, a model that predicts "legit" for
   everything gets 99.83% accuracy — but catches ZERO fraud. Useless.
   We need Precision, Recall, and F1 on the minority class.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def random_oversample(X_train: np.ndarray, y_train: np.ndarray,
                      random_state: int = 42) -> tuple:
    """
    Randomly duplicate minority class (fraud) samples until balanced.

    Pros : Simple, no synthetic data artifacts
    Cons : Overfitting risk (exact copies seen many times)
    """
    rng = np.random.default_rng(random_state)
    fraud_idx = np.where(y_train == 1)[0]
    legit_idx = np.where(y_train == 0)[0]

    n_to_add = len(legit_idx) - len(fraud_idx)
    sampled = rng.choice(fraud_idx, size=n_to_add, replace=True)

    X_resampled = np.vstack([X_train, X_train[sampled]])
    y_resampled = np.concatenate([y_train, y_train[sampled]])

    # Shuffle so model doesn't see all fraud at the end
    idx = rng.permutation(len(y_resampled))
    print(f"✅ Random Oversample: {len(y_train):,} → {len(y_resampled):,} samples "
          f"(fraud: {y_resampled.sum():,})")
    return X_resampled[idx], y_resampled[idx]


def smote_oversample(X_train: np.ndarray, y_train: np.ndarray,
                     k_neighbors: int = 5,
                     sampling_ratio: float = 0.10,
                     random_state: int = 42) -> tuple:
    """
    Synthetic Minority Oversampling Technique (SMOTE).

    Algorithm:
    1. For each fraud sample, find its k nearest fraud neighbors
    2. Randomly pick one neighbor
    3. Create synthetic sample = original + random * (neighbor - original)

    This produces new, unseen fraud examples rather than duplicates.

    Parameters
    ----------
    sampling_ratio : Target fraud/(fraud+legit) ratio after oversampling
    """
    rng = np.random.default_rng(random_state)
    fraud_idx = np.where(y_train == 1)[0]
    legit_idx = np.where(y_train == 0)[0]

    X_fraud = X_train[fraud_idx]
    n_legit = len(legit_idx)
    n_fraud_target = int(n_legit * sampling_ratio / (1 - sampling_ratio))
    n_synthetic = max(0, n_fraud_target - len(fraud_idx))

    if n_synthetic == 0:
        return X_train, y_train

    # Fit KNN on fraud samples only
    k = min(k_neighbors, len(X_fraud) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree')
    nn.fit(X_fraud)
    distances, indices = nn.kneighbors(X_fraud)
    # indices[:,0] is the point itself, so use 1:
    neighbor_indices = indices[:, 1:]

    # Generate synthetic samples
    synthetic_X = []
    for _ in range(n_synthetic):
        # Pick a random fraud sample
        i = rng.integers(0, len(X_fraud))
        # Pick a random neighbor
        nn_idx = rng.integers(0, k)
        neighbor = X_fraud[neighbor_indices[i, nn_idx]]
        # Interpolate
        lam = rng.uniform(0, 1)
        synthetic = X_fraud[i] + lam * (neighbor - X_fraud[i])
        synthetic_X.append(synthetic)

    synthetic_X = np.array(synthetic_X)
    X_resampled = np.vstack([X_train, synthetic_X])
    y_resampled = np.concatenate([y_train, np.ones(n_synthetic, dtype=int)])

    idx = rng.permutation(len(y_resampled))
    print(f"✅ SMOTE: {len(y_train):,} → {len(y_resampled):,} samples "
          f"(fraud: {y_resampled.sum():,}, {y_resampled.mean():.2%})")
    return X_resampled[idx], y_resampled[idx]


def random_undersample(X_train: np.ndarray, y_train: np.ndarray,
                       ratio: float = 10.0,
                       random_state: int = 42) -> tuple:
    """
    Keep all fraud; downsample legit to fraud * ratio.

    Pros : Fast, no artificial data
    Cons : Throws away real information
    """
    rng = np.random.default_rng(random_state)
    fraud_idx = np.where(y_train == 1)[0]
    legit_idx = np.where(y_train == 0)[0]

    n_keep_legit = min(len(legit_idx), int(len(fraud_idx) * ratio))
    kept_legit = rng.choice(legit_idx, size=n_keep_legit, replace=False)

    keep = np.concatenate([fraud_idx, kept_legit])
    rng.shuffle(keep)

    print(f"✅ Undersample: {len(y_train):,} → {len(keep):,} samples "
          f"(fraud: {y_train[keep].sum():,}, {y_train[keep].mean():.2%})")
    return X_train[keep], y_train[keep]
