# Refence: https://syftbox-documentation.openmined.org/cpu-tracker-2

import os
import joblib
from pathlib import Path
from utils.vocab import create_tvseries_vocab
from utils.syftbox import network_participants, create_shared_folder, participants_datasets
from pets.fedavg_mlp import get_users_mlp_parameters, mlp_fedavg
from pets.dp_top5 import dp_top5_series
from pets.phe import generate_keys
from syftbox.lib import Client

API_NAME = os.getenv("API_NAME")
DATA_DIR = os.path.join(os.getcwd(), os.getenv("AGGREGATOR_DATA_DIR"))

def server_initialization(save_to:str, tv_series_path:str, imdb_ratings_path:str):
    import numpy as np
    from participant.server_utils.data_loading import load_tv_vocabulary, load_imdb_ratings
    from participant.federated_learning.svd_server_initialisation import initialize_item_factors

    def normalize_string(s):
        """
        """
        return s.replace('\u200b', '').lower()

    # Step 1: Load vocabulary and IMDB ratings
    tv_vocab = load_tv_vocabulary(tv_series_path)
    imdb_ratings = load_imdb_ratings(imdb_ratings_path)

    # Step 2: Load and normalize IMDB ratings
    imdb_data = np.load(imdb_ratings_path, allow_pickle=True).item()
    imdb_ratings = {normalize_string(title): float(rating) for title, rating in imdb_data.items() if rating}

    # Step 2: Initialize item factors
    V = initialize_item_factors(tv_vocab, imdb_ratings)

    # Step 4: Save the initialized model
    os.makedirs(save_to, exist_ok=True)
    np.save(os.path.join(save_to, "global_V.npy"), V)

    print("Server initialization complete. Item factors (V) are saved.")

def server_aggregate(updates, save_to, weights=None, learning_rate=1.0, epsilon=1.0, clipping_threshold=0.5):
    """
    Orchestrates the server aggregation process:
    1. Loads current global item factors.
    2. Calls `aggregate_item_factors` to perform the aggregation.
    3. Saves the updated global item factors.

    Args:
        updates (list[dict]): List of delta dictionaries from participants.
        weights (list[float]): List of weights for each participant. If None, equal weights are assumed.
        learning_rate (float): Scaling factor for the aggregated deltas.
        epsilon (float): Privacy budget for differential privacy.
        clipping_threshold (float): Clipping threshold for updates.
        save_to (str): Path to save the updated global item factors.
    """
    import numpy as np
    from participant.server_utils.data_loading import load_global_item_factors
    from participant.federated_learning.svd_server_aggregation import aggregate_item_factors
    global_V_path = os.path.join(save_to, "global_V.npy")

    # Step 1: Load current global item factors
    V = load_global_item_factors(global_V_path)

    # Step 2: Aggregate updates
    V = aggregate_item_factors(
        V, updates, weights=weights, learning_rate=learning_rate, epsilon=epsilon, clipping_threshold=clipping_threshold
    )

    # Step 3: Save the updated global item factors
    os.makedirs(os.path.dirname(global_V_path), exist_ok=True)
    np.save(global_V_path, V)

    print("Server aggregation complete. Global item factors (V) updated.")

if __name__ == "__main__":
    client = Client.load()

    datasites_path = Path(client.datasite_path.parent)   # automatically retrieve datasites path

    peers = network_participants(datasites_path, API_NAME)         # check participant of netflix trend
    peers_w_netflix_data = participants_datasets(datasites_path, dataset_name = "Netflix Data", dataset_format = "CSV")  # check for "Netflix Data" from datasites/<user>/public/datasets.yaml

    print(f"[!] Participants with the App Installed: {peers}")
    print(f"[!] Participants with Netflix Data but not with the App Installed: {[peer for peer in peers_w_netflix_data if peer not in peers]}")

    # Here we do not use public folder for aggregator, but an api_folder accesible to participants only
    shared_folder_path = create_shared_folder(Path(client.datasite_path), API_NAME, client, peers)

    # Paillier Homomorphic Encryption Setup
    private_path = client.datasite_path / "private" / API_NAME
    generate_keys(public_path=shared_folder_path, private_path=private_path)
    
    # Create a Vocabulary of TV Series
    create_tvseries_vocab(shared_folder_path)
    
    # MLP use case -> FedAvg
    weights, biases = get_users_mlp_parameters(datasites_path, API_NAME, peers)    # MLP: retrieve the path to weights and bias
    try:
        fedavg_weights, fedavg_biases = mlp_fedavg(weights, biases)
        joblib.dump(fedavg_weights, shared_folder_path / "netflix_mlp_fedavg_weights.joblib")
        joblib.dump(fedavg_biases, shared_folder_path / "netflix_mlp_fedavg_biases.joblib")
    except Exception as e:
        print(f"> Error to perform FedAvg: {e}")

    # Differential Privacy use case -> Top-5 Most Seen TV Series
    MIN_PARTICIPANTS = 3
    if len(peers) > MIN_PARTICIPANTS:  # check the top-5 if at least MIN_PARTICIPANTS available
        dp_top5_series(datasites_path, peers, min_participants=MIN_PARTICIPANTS)
        # TODO: update assets -> static index

    # Check if global V exists
    if not os.path.exists(shared_folder_path / "global_V.npy"):
        server_initialization(save_to=shared_folder_path, 
                            tv_series_path=shared_folder_path / "tv-series_vocabulary.json", 
                            imdb_ratings_path="data/imdb_ratings.npy")