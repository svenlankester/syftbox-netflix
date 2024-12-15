import os
import json
import numpy as np

def load_tv_vocabulary(vocabulary_path):
    """
    Load the TV series vocabulary from the specified JSON file.
    """
    with open(vocabulary_path, "r") as f:
        return json.load(f)

def load_participant_ratings(private_folder):
    """
    Load participant's ratings from the private folder.
    """
    ratings_path = os.path.join(private_folder, "ratings.npy")
    return np.load(ratings_path, allow_pickle=True).item()

def load_global_item_factors(save_path):
    """
    Load the global item factors matrix (V).
    """
    global_V_path = os.path.join(save_path, "global_V.npy")
    return np.load(global_V_path)

def load_or_initialize_user_matrix(user_id, latent_dim, save_path="mock_dataset_location/tmp_model_parms"):

    user_matrix_path = os.path.join(save_path, f"{user_id}_U.npy")
    if os.path.exists(user_matrix_path):
        U_u = np.load(user_matrix_path)
        print(f"Loaded existing user matrix for {user_id}.")
    else:
        U_u = initialize_user_matrix(user_id, latent_dim, save_path)
    return U_u

def initialize_user_matrix(user_id, latent_dim, save_path="mock_dataset_location/tmp_model_parms"):

    # Create save directory if not exists
    os.makedirs(save_path, exist_ok=True)

    # Initialize user matrix
    U_u = np.random.normal(scale=0.01, size=(latent_dim,))

    # Save user matrix
    user_matrix_path = os.path.join(save_path, f"{user_id}_U.npy")
    np.save(user_matrix_path, U_u)
    print(f"Initialized and saved user matrix for {user_id}.")
    return U_u
