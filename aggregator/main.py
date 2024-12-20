# Refence: https://syftbox-documentation.openmined.org/cpu-tracker-2

import os
import numpy as np
import joblib
import logging
from pathlib import Path
from dotenv import load_dotenv
from utils.vocab import create_tvseries_vocab
from utils.frontend import populate_html_template
from utils.syftbox import network_participants, create_shared_folder, participants_datasets
from pets.svd_aggregator import svd_engine_init_and_aggregate
from pets.svd_recommender import local_recommendation
from pets.dp_top5 import dp_top5_series
from pets.phe import generate_keys
from syftbox.lib import Client

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ANSI escape codes for colors
COLORS = {
    "DEBUG": "\033[94m",  # Blue
    "INFO": "\033[92m",   # Green
    "WARNING": "\033[93m",  # Yellow
    "ERROR": "\033[91m",   # Red
    "RESET": "\033[0m",    # Reset
}

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        color = COLORS.get(record.levelname, COLORS["RESET"])
        reset = COLORS["RESET"]
        record.msg = f"{color}{record.msg}{reset}"
        return super().format(record)

# Apply the colored formatter to all handlers
for handler in logging.getLogger().handlers:
    handler.setFormatter(ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s"))

# Load environment variables
load_dotenv()
API_NAME = os.getenv("API_NAME")
DATA_DIR = os.path.join(os.getcwd(), os.getenv("AGGREGATOR_DATA_DIR"))
svd_init = False


if __name__ == "__main__":
    try:
        client = Client.load()
        logging.info("Client loaded successfully.")

        datasites_path = Path(client.datasite_path.parent)
        logging.debug("Retrieving datasites path...")

        peers = network_participants(datasites_path, API_NAME)
        peers_w_netflix_data = participants_datasets(
            datasites_path, dataset_name="Netflix Data", dataset_format="CSV"
        )
        logging.info(f"[!] Participants with the App Installed: {peers}")
        logging.info(f"[!] Participants with Netflix Data but not the App Installed: "
                     f"{[peer for peer in peers_w_netflix_data if peer not in peers]}")

        # Shared folder setup
        shared_folder_path = create_shared_folder(Path(client.datasite_path), API_NAME, client, peers)
        logging.debug("Shared folder created successfully.")

        # Paillier Homomorphic Encryption setup
        private_path = client.datasite_path / "private" / API_NAME
        generate_keys(public_path=shared_folder_path, private_path=private_path)
        logging.info("Paillier Homomorphic Encryption keys generated.")

        # Create a vocabulary of TV series
        tv_vocab = create_tvseries_vocab(shared_folder_path)
        logging.debug("TV series vocabulary created.")

        # Initialize the SVD model and aggregate the available deltas for fine-tuning
        svd_engine_init_and_aggregate(datasites_path, shared_folder_path, API_NAME, peers, svd_init)

        # Differential Privacy Top-5 TV Series
        MIN_PARTICIPANTS = 3
        try:
            logging.debug("Performing differential privacy Top-5 TV series computation...")
            available_dp_vectors = dp_top5_series(datasites_path, peers, min_participants=MIN_PARTICIPANTS)
            logging.info("Top-5 Differential Privacy Computation Completed Successfully.")

            top_series_path: Path = ( client.datasite_path / "private" / API_NAME / "top5_series.json")
            template_path = Path("./aggregator/assets/top5-series.html")
            output_path = client.datasite_path / "index.html"

        except Exception as e:
            logging.error(f"An unexpected error occurred during DP Top-5 computation: {e}")
        
        try:
            logging.debug("Performing Local SVD Recommandations...")
            ## Note, this is a local recommendation, not a federated one
            ## Only works for cases where the aggregator account is also a participant, and uses the aggregator's private data to generate recommendations
            ## This is for DEMO purposes while maintaining privacy of the participants
            participant_private_path = client.datasite_path / "private" / API_NAME / 'profile_0'
            recommendations = local_recommendation(participant_private_path, shared_folder_path, tv_vocab, exclude_watched=True)
        except Exception as e:
            logging.error(f"An unexpected error occurred during Local SVD Recommandations: {e}")
            recommendations = None

        # Populate the HTML template with the Top 5 Community Shows and Personalised Recommendations
        if available_dp_vectors > MIN_PARTICIPANTS:
            populate_html_template(top_series_path, template_path, output_path, available_dp_vectors, recommendations=recommendations)

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
