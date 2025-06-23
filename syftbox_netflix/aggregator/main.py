# Refence: https://syftbox-documentation.openmined.org/cpu-tracker-2

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from syftbox_netflix.aggregator.pets.dp_top5 import dp_top5_series
from syftbox_netflix.aggregator.pets.phe import generate_keys
from syftbox_netflix.aggregator.pets.svd_aggregator import svd_engine_init_and_aggregate
from syftbox_netflix.aggregator.pets.svd_recommender import local_recommendation
from syft_core import Client as SyftboxClient
from syft_core import SyftClientConfig
from syftbox_netflix.aggregator.utils.frontend import populate_html_template
from syftbox_netflix.aggregator.utils.syftbox import (
    create_shared_folder,
    create_recommendation_dataset,
    network_participants,
    participants_datasets,
)
from syftbox_netflix.aggregator.utils.vocab import create_tvseries_vocab

# Load environment variables
load_dotenv()
APP_NAME = os.getenv("APP_NAME", "syftbox-netflix")
DATA_DIR = os.path.join(os.getcwd(), os.getenv("AGGREGATOR_DATA_DIR"))
svd_init = False


def main():
    try:
        config = SyftClientConfig.load()
        client = SyftboxClient(config)
        SYFTBOX_PATH = Path(client.config.data_dir)

        logging.info("Client loaded successfully.")

        datasites_path = Path(client.datasite_path.parent)
        logging.debug("Retrieving datasites path...")

        peers = network_participants(datasites_path, APP_NAME)
        peers_w_netflix_data = participants_datasets(
            datasites_path, dataset_name="Netflix Data", dataset_format="CSV"
        )
        logging.info(f"[!] Participants with the App Installed: {peers}")
        logging.info(
            f"[!] Participants with Netflix Data but not the App Installed: "
            f"{[peer for peer in peers_w_netflix_data if peer not in peers]}"
        )

        # Shared folder setup
        shared_folder_path = create_shared_folder(
            Path(client.datasite_path), APP_NAME, client, peers
        )
        logging.debug("Shared folder created successfully.")

        logging.debug("Creating recommendation dataset...")
        _ = create_recommendation_dataset(Path(client.datasite_path), APP_NAME, client, peers)

        # Paillier Homomorphic Encryption setup
        private_path = SYFTBOX_PATH / "private" / APP_NAME
        generate_keys(public_path=shared_folder_path, private_path=private_path)
        logging.info(f"Paillier Homomorphic Encryption keys generated.")

        # Create a vocabulary of TV series
        tv_vocab = create_tvseries_vocab(shared_folder_path)
        logging.debug("TV series vocabulary created.")

        # Initialize the SVD model and aggregate the available deltas for fine-tuning
        svd_engine_init_and_aggregate(
            datasites_path, shared_folder_path, APP_NAME, peers, svd_init
        )

        # Differential Privacy Top-5 TV Series
        MIN_PARTICIPANTS = 3
        try:
            logging.debug(
                "Performing differential privacy Top-5 TV series computation..."
            )
            available_dp_vectors = dp_top5_series(
                datasites_path, peers, min_participants=MIN_PARTICIPANTS
            )
            logging.info(
                "Top-5 Differential Privacy Computation Completed Successfully."
            )

            top_series_path: Path = (
                client.datasite_path / "app_data" / APP_NAME / "shared" / "top5_series.json"
            )
            template_path = os.path.join(os.getcwd(), "syftbox_netflix", "aggregator", "assets", "top5-series.html")
            
            output_path = client.datasite_path / "public" / "index.html"

        except Exception as e:
            logging.error(
                f"An unexpected error occurred during DP Top-5 computation: {e}"
            )

        try:
            logging.debug("Performing Local SVD Recommandations...")
            ## Note, this is a local recommendation, not a federated one
            ## Only works for cases where the aggregator account is also a participant, and uses the aggregator's private data to generate recommendations
            ## This is for DEMO purposes while maintaining privacy of the participants
            participant_private_path = (
                SYFTBOX_PATH / "private" / APP_NAME / "profile_0"
            )
            recommendations = local_recommendation(
                participant_private_path,
                shared_folder_path,
                tv_vocab,
                exclude_watched=True,
            )
        except Exception as e:
            logging.error(
                f"An unexpected error occurred during Local SVD Recommandations: {e}"
            )
            recommendations = None

        # Populate the HTML template with the Top 5 Community Shows and Personalised Recommendations
        if available_dp_vectors > MIN_PARTICIPANTS:
            populate_html_template(
                top_series_path,
                template_path,
                output_path,
                available_dp_vectors,
                recommendations=recommendations,
            )

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
