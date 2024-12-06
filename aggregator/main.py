# Refence: https://syftbox-documentation.openmined.org/cpu-tracker-2

import os
import joblib
from pathlib import Path
from utils.checks import should_run
from utils.vocab import create_tvseries_vocab
from utils.syftbox import network_participants, create_shared_folder
from pets.fedavg_mlp import get_users_mlp_parameters, mlp_fedavg
from pets.dp_top5 import dp_top5_series
from syftbox.lib import Client


API_NAME = os.getenv("API_NAME")
DATA_DIR = os.path.join(os.getcwd(), os.getenv("AGGREGATOR_DATA_DIR"))

if __name__ == "__main__":
    client = Client.load()

    if not should_run(60):
        print(f"Skipping {API_NAME} as Aggregator, not enough time has passed.")
        exit(0)

    datasites_path = Path(client.datasite_path.parent)   # automatically retrieve datasites path

    peers = network_participants(datasites_path)         # check participant of netflix trend

    # Here we do not use public folder for aggregator, but an api_folder accesible to participants only
    shared_folder_path = create_shared_folder(Path(client.datasite_path), client, peers)
    
    # Create a Vocabulary of TV Series
    create_tvseries_vocab(shared_folder_path)
    
    # MLP use case -> FedAvg
    weights, biases = get_users_mlp_parameters(datasites_path, peers)    # MLP: retrieve the path to weights and bias
    fedavg_weights, fedavg_biases = mlp_fedavg(weights, biases)
    joblib.dump(fedavg_weights, shared_folder_path / "netflix_mlp_fedavg_weights.joblib")
    joblib.dump(fedavg_biases, shared_folder_path / "netflix_mlp_fedavg_biases.joblib")

    # Differential Privacy use case -> Top-5 Most Seen TV Series
    MIN_PARTICIPANTS = 3
    if len(peers) > MIN_PARTICIPANTS:  # check the top-5 if at least MIN_PARTICIPANTS available
        dp_top5_series(datasites_path, peers, min_participants=MIN_PARTICIPANTS)
        # TODO: update assets -> static index