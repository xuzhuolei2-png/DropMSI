# -*- coding: utf-8 -*-

import os

import numpy as np
from scipy.sparse import load_npz
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# Preprocessing output filenames
# ============================================================

NORMALIZATION_FILENAME = "normalization.npy"

MISSING_FILENAME = "training_observation.npy"

MASK_FILENAME = "0.1_mask.npy"

BACKGROUND_FILENAME = "tissue_mask.npy"

Q_FILENAME = "imputation_keep_mask.npy"


def preprocess_data(
    input_npz,
    output_dir,
    random_seed=42,
    mask_rate=0.1,
    percentile=99.9,
    n_clusters=2,
    kmeans_random_state=0,
    background_threshold=1e-3
):
    """
    Preprocess the input MSI matrix.

    Parameters
    ----------
    input_npz : Path to the input sparse matrix NPZ file.

    output_dir : Directory for saving the preprocessing results.

    random_seed :default=42

    mask_rate : float, default=0.1
        Random masking probability.

    percentile : float, default=99.9
        Percentile clipping parameter after TIC normalization.

    n_clusters : int, default=2
        Number of KMeans clusters used for background identification.

    kmeans_random_state : int, default=0
        The random_state parameter of KMeans.

    background_threshold : float, default=1e-3
        Threshold compensation parameter for background identification.
    """

    input_npz = os.fspath(input_npz)
    output_dir = os.fspath(output_dir)

    if not os.path.isfile(input_npz):
        raise FileNotFoundError(
            f"Input NPZ file does not exist: {input_npz}"
        )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # Random seed
    np.random.seed(random_seed)

    print("=" * 60)
    print("Starting preprocessing")
    print(f"Input NPZ file: {input_npz}")
    print(f"Preprocessing output directory: {output_dir}")
    print(f"Random seed: {random_seed}")
    print(f"Artificial masking rate: {mask_rate}")
    print("=" * 60)

    # ========================================================
    # 1. Define output file paths
    # ========================================================

    normalization_path = os.path.join(
        output_dir,
        NORMALIZATION_FILENAME
    )

    missing_path = os.path.join(
        output_dir,
        MISSING_FILENAME
    )

    mask_path = os.path.join(
        output_dir,
        MASK_FILENAME
    )

    background_path = os.path.join(
        output_dir,
        BACKGROUND_FILENAME
    )

    q_path = os.path.join(
        output_dir,
        Q_FILENAME
    )

    # ========================================================
    # 2. Load and normalize the data
    # ========================================================

    sparse_matrix = load_npz(input_npz)

    data = sparse_matrix.toarray().astype( "float32")

    num_nodes, num_features = data.shape

    print(
        f"Input data shape: "
        f"number of nodes={num_nodes}, "
        f"number of features={num_features}"
    )

    # --------------------------------------------------------
    # TIC normalization
    # --------------------------------------------------------

    data_sum = np.sum(data,axis=1).reshape(-1, 1)

    data_sum = np.where(data_sum == 0,1,data_sum)

    data_tic = (data / data_sum)

    # --------------------------------------------------------
    # Percentile clipping
    # --------------------------------------------------------

    percentile_bound = np.percentile(
        data_tic,
        percentile,
        axis=0
    )

    norm_data = np.zeros_like(
        data_tic
    )

    for feature_idx in range(
        data_tic.shape[1]
    ):
        current_feature = data_tic[
            :,
            feature_idx
        ]

        norm_data[
            :,
            feature_idx
        ] = np.where(
            current_feature
            > percentile_bound[feature_idx],
            percentile_bound[feature_idx],
            current_feature
        )

    # --------------------------------------------------------
    # Min-Max normalization
    # --------------------------------------------------------

    norm_data = MinMaxScaler().fit_transform(
        norm_data
    )

    np.save(
        normalization_path,
        norm_data
    )

    print(
        f"Saved: {NORMALIZATION_FILENAME} "
        f"{norm_data.shape}"
    )

    # ========================================================
    # 3. Record observed positions
    # ========================================================

    data_array_T = norm_data.T

    original_missing = (
        data_array_T > 0
    ).astype("float32")

    # ========================================================
    # 4. Random masking
    # ========================================================

    mask_gate = np.random.rand(
        *original_missing.shape
    )

    mask_matrix = np.ones_like(
        original_missing
    )

    mask_condition = (
        (original_missing == 1)
        & (mask_gate < mask_rate)
    )

    mask_matrix[
        mask_condition
    ] = 0

    masked_missing = (
        original_missing
        * mask_matrix
    )

    np.save(
        missing_path,
        masked_missing
    )

    np.save(
        mask_path,
        mask_matrix
    )

    print(
        f"Saved: {MISSING_FILENAME} & "
        f"{MASK_FILENAME} "
        f"{masked_missing.shape}"
    )

    # ========================================================
    # 5. Identify the background
    # ========================================================

    max_val_per_pixel = np.max(
        data,
        axis=1
    ).reshape(-1, 1)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=kmeans_random_state
    )

    cluster_labels = kmeans.fit_predict(
        max_val_per_pixel
    )

    cluster_centers = (
        kmeans.cluster_centers_.flatten()
    )

    background_cluster = np.argmin(
        np.abs(cluster_centers)
    )

    first_column = np.where(
        cluster_labels == background_cluster,
        0,
        1
    )

    # ========================================================
    # 6. Threshold compensation
    # ========================================================

    first_column[
        (
            max_val_per_pixel.flatten()
            > background_threshold
        )
        & (first_column == 0)
    ] = 1

    first_column = first_column.reshape(
        -1,
        1
    )

    np.save(
        background_path,
        first_column
    )

    print(
        f"Saved: {BACKGROUND_FILENAME} "
        f"{first_column.shape}"
    )

    # ========================================================
    # 7. Generate the imputation control matrix
    # ========================================================

    q3_data = masked_missing.copy()

    background_columns = np.where(
        first_column.flatten() == 0
    )[0]

    q3_data[
        :,
        background_columns
    ] = 1

    np.save(
        q_path,
        q3_data
    )

    print(
        f"Saved: {Q_FILENAME} "
        f"{q3_data.shape}"
    )

    print()
    print("=" * 60)
    print("Preprocessing completed successfully!")
    print(
        f"All .npy files have been saved to: "
        f"{output_dir}"
    )
    print("=" * 60)

    # ========================================================
    # Return results to run.py
    # ========================================================

    preprocessing_result = {
        "input_npz": input_npz,
        "output_dir": output_dir,
        "data_shape": tuple(data.shape),
        "num_nodes": num_nodes,
        "num_features": num_features,
        "normalization_path": normalization_path,
        "missing_path": missing_path,
        "mask_path": mask_path,
        "background_path": background_path,
        "q_path": q_path,
    }

    return preprocessing_result


__all__ = [
    "preprocess_data",
    "NORMALIZATION_FILENAME",
    "MISSING_FILENAME",
    "MASK_FILENAME",
    "BACKGROUND_FILENAME",
    "Q_FILENAME",
]