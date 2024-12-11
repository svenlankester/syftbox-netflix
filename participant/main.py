import os
import sys
import numpy as np
import csv
from typing import Tuple
from datetime import datetime
from pathlib import Path
import subprocess
from syftbox.lib import Client, SyftPermission
from participant_utils.checks import should_run

# Package functions
from loaders.netflix_loader import download_daily_data, get_latest_file
import federated_analytics.data_processing as fa
import federated_learning.mlp_model as mlp
from federated_learning.sequence_data import SequenceData
from federated_learning.sequence_data import create_view_counts_vector
from federated_analytics.dp_series import run_top5_dp

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
    file_path_static = os.path.join(datapath, netflix_csv_prefix + ".csv")

    try:
        # Try to download the file using Chromedriver
        try:
            chromedriver_path = subprocess.check_output(['which', 'chromedriver'], text=True).strip()
            os.environ['CHROMEDRIVER_PATH'] = chromedriver_path
            if not os.path.exists(file_path):
                print(f"Data file not found. Downloading to {file_path}...")
                download_daily_data(datapath, filename)
                print(f"Successfully downloaded Netflix data to {file_path}.")
            static_file = False
            
        except subprocess.CalledProcessError:
            print(f">> ChromeDriver not found. Unable to retrieve from Netflix via download.")
            print(f"Checking for a locally available static file: {file_path_static}...")
            
        except Exception as e:
            print(f"{e}")

            # Try to use the static file if downloading failed
            if os.path.exists(file_path_static):
                print(f"Using static viewing history (manually downloaded from Netflix): {file_path_static}...")
                static_file = True
            else:
                print((
                    f">> Neither ChromeDriver is available for download nor the static file exists. "
                    f"Please retrieve the file manually from Netflix and make it available here: \n\t\t {datapath}"
                ))
            
                print(f">> Copying dummy file (data/dummy.csv) to {file_path_static}.")
                try:
                    with open('data/dummy.csv', 'rb') as src_file:
                        with open(file_path_static, 'wb') as dest_file:
                            dest_file.write(src_file.read())
                    print(f">> Copied dummy file (data/dummy.csv) to {file_path_static}. For test purpose only!")
                    static_file = True
                except Exception as e: 
                    print(f"[!] Error copying dummy file (data/dummy.csv) to {file_path_static}: {e}")
                    sys.exit(1)

    except Exception as e:
        print(f"Error retrieving Netflix data: {e}")
        raise

    if static_file:
        latest_data_file = file_path_static
    else:
        latest_data_file = get_latest_file(datapath, csv_name)

    # Load the CSV into a NumPy array
    print(f"Loading data from {latest_data_file}...")
    return latest_data_file, load_csv_to_numpy(latest_data_file)

def run_federated_analytics(restricted_public_folder, private_folder, viewing_history):
    # Reduce and aggregate the original information
    reduced_history = fa.orchestrate_reduction(viewing_history)
    aggregated_history = fa.aggregate_title_week_counts(reduced_history)

    # For Debugging: Filter rows where index 1 is "Avatar"
    # filtered_rows = aggregated_history[aggregated_history[:, 0] == 'Avatar']

    # Infer ratings as per viewing patterns
    ratings_dict = fa.calculate_show_ratings(aggregated_history)

    netflix_file_path = 'data/netflix_titles.csv'
    netflix_show_data = load_csv_to_numpy(netflix_file_path)

    title_genre_dict = fa.create_title_genre_dict(netflix_show_data, title_col=2, genre_col=10) # tmp dict - may be useful for aggregates.
    user_information = fa.add_column_from_dict(aggregated_history, ratings_dict, key_col=0, new_col_name='rating')

    # This is an enhanced data compared with the retrieved viewing history from Netflix website
    # Useful for more complex analytics
    my_shows_data = fa.join_viewing_history_with_netflix(user_information, netflix_show_data)

    # Save data
    fa.save_npy_data(restricted_public_folder, "netflix_reduced.npy", reduced_history)
    fa.save_npy_data(restricted_public_folder, "netflix_aggregated.npy", aggregated_history)
    fa.save_npy_data(private_folder, "netflix_full.npy", viewing_history)
    fa.save_npy_data(private_folder, "my_shows_data_full.npy", my_shows_data)
    fa.save_npy_data(private_folder, "my_shows_data_ratings.npy", ratings_dict)

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
    run_top5_dp(private_folder / "tvseries_views_sparse_vector.npy", restricted_public_folder, verbose=False)
    ##############

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        sys.exit(1)



