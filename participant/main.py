import os
import sys
import numpy as np
import csv
from typing import Tuple
from datetime import datetime
from pathlib import Path
from syftbox.lib import Client, SyftPermission
from participant_utils.checks import should_run

# Package functions
from loaders.netflix_loader import download_daily_data, get_latest_file
import federated_analytics.data_processing as fa
import federated_learning.mlp_model as mlp
from federated_learning.sequence_data import SequenceData
from federated_learning.sequence_data import create_view_counts_vector

from dotenv import load_dotenv
load_dotenv()
API_NAME = os.getenv("API_NAME")
AGGREGATOR_DATASITE = os.getenv("AGGREGATOR_DATASITE")
CSV_NAME = os.getenv("NETFLIX_CSV")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

def load_csv_to_numpy(file_path: str) -> np.ndarray:
    """
    Load a CSV file into a NumPy array, handling quoted fields.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        np.ndarray: A 2D NumPy array containing the data from the CSV.
    """
    cleaned_data = []

    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            cleaned_data.append(row)

    return np.array(cleaned_data)

def setup_environment(client, api_name, aggregator_path):
    """
    Set up public and private folders for data storage.

    Args:
        client: Client instance for managing API and datasite paths.

    Returns:
        tuple: Paths to restricted public and private folders.
    """

    def create_private_folder(path: Path, client: Client) -> Path:
        """
        Create a private folder within the specified path.

        This function creates a directory structure containing the NetflixViewingHistory.csv.
        """

        netflix_datapath: Path = path / "private" / "netflix_data"
        os.makedirs(netflix_datapath, exist_ok=True)

        # Set the default permissions
        permissions = SyftPermission.datasite_default(email=client.email)
        permissions.save(netflix_datapath)  # update the ._syftperm

        return netflix_datapath

    def create_public_folder(path: Path, client: Client, aggregator_path) -> None:
        """
        Create a API public folder within the specified path.

        This function creates a directory for receiving the private enhanced version \
        of the viewing history.
        """

        os.makedirs(path, exist_ok=True)

        # Set default permissions for this folder
        permissions = SyftPermission.datasite_default(email=client.email)
        permissions.read.append(aggregator_path) # set read permission to the aggregator
        permissions.save(path)

    restricted_public_folder = client.api_data(api_name)
    create_public_folder(restricted_public_folder, client, aggregator_path)
    private_folder = create_private_folder(client.datasite_path, client)
    return restricted_public_folder, private_folder

def get_or_download_latest_data(output_dir, csv_name) -> Tuple[str, np.ndarray]:
    """
    Ensure the latest Netflix data exists or download it if missing.
    After retrieval, load the data into a NumPy array for further processing.

    Returns:
        np.ndarray: The latest Netflix viewing history as a structured array.
    """
    # Construct paths and file names
    datapath = os.path.expanduser(output_dir)
    today_date = datetime.now().strftime("%Y-%m-%d")
    netflix_csv_prefix = os.path.splitext(csv_name)[0]
    filename = f"{netflix_csv_prefix}_{today_date}.csv"
    file_path = os.path.join(datapath, filename)

    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Data file not found. Downloading to {file_path}...")
            download_daily_data(datapath, filename)

    except Exception as e:
        print(f"Error retrieving Netflix data: {e}")
        raise

    latest_data_file = get_latest_file(datapath, csv_name)

    # Load the CSV into a NumPy array
    print(f"Loading data from {latest_data_file}...")
    return latest_data_file, load_csv_to_numpy(latest_data_file)

def run_federated_analytics(restricted_public_folder, private_folder, viewing_history):
    # Reduce the original information
    reduced_history = fa.orchestrate_reduction(viewing_history)

    netflix_file_path = 'data/netflix_titles.csv'
    netflix_show_data = load_csv_to_numpy(netflix_file_path)

    # This is an enhanced data compared with the retrieved viewing history from Netflix website
    # Useful for more complex analytics
    my_shows_data = fa.join_viewing_history_with_netflix(reduced_history, netflix_show_data)

    # Process and save watch history
    fa.aggregate_and_store_history(
        reduced_history, 
        viewing_history, 
        private_folder, 
        restricted_public_folder
    )

    private_shows_file: Path = private_folder / "my_shows_data_full.npy"
    np.save(str(private_shows_file), my_shows_data)

def run_federated_learning(aggregator_path, restricted_public_folder, private_folder, viewing_history, latest_data_file, datasite_parent_path):
    netflix_file_path = 'data/netflix_titles.csv'
    netflix_show_data = load_csv_to_numpy(netflix_file_path)

    # Useful for Embeddings and more complex learning
    my_shows_data = fa.join_viewing_history_with_netflix(viewing_history, netflix_show_data)

    # Train and save MLP model
    mlp.train_and_save_mlp(latest_data_file, restricted_public_folder)

    # Create a sequence data (filter by > 1 episodes)
    # Columns: series (TV series title), Total_Views (quantity), First_Seen (datetime)
    # - loaded with the original NetflixViewingHistory.csv
    sequence_recommender = SequenceData(viewing_history)
    
    view_counts_vector = create_view_counts_vector(aggregator_path, sequence_recommender.aggregated_data, datasite_parent_path)
    private_tvseries_views_file: Path = private_folder / "tvseries_views_sparse_vector.npy"
    np.save(str(private_tvseries_views_file), view_counts_vector)

def main():

    client = Client.load()

    # Skip execution if conditions are not met
    if not should_run():
        print(f"Skipping {API_NAME} as Participant, not enough time has passed.")
        sys.exit(0)

    # Set up environment
    restricted_public_folder, private_folder = setup_environment(client, API_NAME, AGGREGATOR_DATASITE)

    # Fetch and load Netflix data
    latest_data_file, viewing_history = get_or_download_latest_data(OUTPUT_DIR, CSV_NAME)

    # Run private processes and write to public/private/restricted directories
    run_federated_analytics(restricted_public_folder, private_folder, viewing_history)
    run_federated_learning(AGGREGATOR_DATASITE, restricted_public_folder, private_folder, viewing_history, latest_data_file, client.datasite_path.parent)

    ##############

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        sys.exit(1)



