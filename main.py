import os
import sys
import csv
import joblib
import json
import numpy as np
import pandas as pd
import subprocess
from typing import Tuple
from rapidfuzz import process
from datetime import datetime
from pathlib import Path
from syftbox.lib import Client, SyftPermission
from collections import Counter
from dotenv import load_dotenv
from fetcher import NetflixFetcher
from utils.ml import train_model, SequenceData
from utils.checks import is_file_modified_today, should_run

# Load environment variables
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


## ==================================================================================================
## Data Processing (1) - Reduction
## ==================================================================================================

def extract_titles(history: np.ndarray) -> np.ndarray:
    """
    Extract and reduce titles from the viewing history.
    """
    return np.array([title.split(":")[0] if ":" in title else title for title in history[:, 0]])


def convert_dates_to_weeks(history: np.ndarray) -> np.ndarray:
    """
    Convert viewing dates to ISO week numbers.
    """
    return np.array([
        datetime.strptime(date, "%d/%m/%Y").isocalendar()[1]
        for date in history[:, 1]
    ])

def orchestrate_reduction(history: np.ndarray) -> np.ndarray:
    """
    Orchestrates the reduction process for Netflix viewing history.
    """
    titles = extract_titles(history)
    weeks = convert_dates_to_weeks(history)
    return np.column_stack((titles, weeks))


## ==================================================================================================
## Netflix Loader functions
## ==================================================================================================

def download_daily_data(output_dir:str, file_name:str) -> None:
    """
    Download Netflix data into today's subfolder.
    """
    downloader = NetflixFetcher(output_dir)
    downloader.run()

    # Validate the file exists after download
    file_path = os.path.join(output_dir, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Netflix viewing history file was not created: {file_path}")

def get_latest_file(subfolder_path):
    """
    Get the latest file in the subfolder by datetime in filename.
    """
    netflix_csv_prefix = os.path.splitext(CSV_NAME)[0] + "_"

    # List all relevant CSV files in the subfolder
    files = [
        f for f in os.listdir(subfolder_path)
        if os.path.isfile(os.path.join(subfolder_path, f)) and f.startswith(netflix_csv_prefix)
    ]

    if not files:
        raise FileNotFoundError(f"No files found in {subfolder_path}")

    # Extract dates and sort files by date descending
    def extract_datetime(filename):
        try:
            date_str = filename.replace(netflix_csv_prefix, "").replace(".csv", "")
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return None

    files_with_dates = [(f, extract_datetime(f)) for f in files]
    valid_files = [(f, dt) for f, dt in files_with_dates if dt is not None]

    if not valid_files:
        raise FileNotFoundError(f"No valid files with dates found in {subfolder_path}")

    latest_file = max(valid_files, key=lambda x: x[1])[0]
    return os.path.join(subfolder_path, latest_file)

def get_or_download_latest_data() -> Tuple[str, np.ndarray]:
    """
    Ensure the latest Netflix data exists or download it if missing.
    After retrieval, load the data into a NumPy array for further processing.

    Returns:
        np.ndarray: The latest Netflix viewing history as a structured array.
    """
    # Construct paths and file names
    datapath = os.path.expanduser(OUTPUT_DIR)
    today_date = datetime.now().strftime("%Y-%m-%d")
    netflix_csv_prefix = os.path.splitext(CSV_NAME)[0]
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

    latest_data_file = get_latest_file(datapath)

    # Load the CSV into a NumPy array
    print(f"Loading data from {latest_data_file}...")
    return latest_data_file, load_csv_to_numpy(latest_data_file)

## ==================================================================================================
## Data Processing (2) - Data Enrichment (shows) and view count vectors
## ==================================================================================================

def join_viewing_history_with_netflix(reduced_history, netflix_show_data):
    """
    Join the reduced viewing history with Netflix show data based on titles.

    Args:
        reduced_history (np.ndarray): Viewing history data.
        netflix_show_data (np.ndarray): Netflix titles data.

    Returns:
        list: Joined data as a list of dictionaries with keys from both datasets.
    """
    # Reduce the viewing history to relevant titles
    my_titles = set([str(title) for title in reduced_history[:, 0]])

    # Get Netflix show titles
    netflix_titles_dict = {
        str(row[2]): row for row in netflix_show_data
    }  # Use title as key for efficient lookups

    # Perform an inner join
    joined_rows = []

    for row in reduced_history:
        title = str(row[0])
        if title in netflix_titles_dict:
            netflix_row = netflix_titles_dict[title]
            # Combine viewing history row and Netflix data row
            joined_rows.append(np.concatenate((row, netflix_row)))

    joined_data = np.array(joined_rows)
    my_found_titles = set([str(title) for title in joined_data[:, 0]])

    print(f"Found Titles: {len(my_found_titles)}")
    print(f"Not Found Titles: {len(my_titles) - len(my_found_titles)}")

    return joined_data

def match_title(title, vocabulary: dict, threshold=80):
    # Exact match
    if title in vocabulary:
        return vocabulary[title]
    
    # Fuzzy match
    vocab_keys = list(vocabulary.keys())  # Convert keys to list for fuzzy matching
    match_result = process.extractOne(title, vocab_keys)
    
    # Extract only the best match and score
    if match_result is not None:
        best_match, score = match_result[:2]  # Unpack the first two values
        if score >= threshold:
            return vocabulary[best_match]
    
    # If no match, return -1
    return -1

def create_view_counts_vector(aggregated_data: pd.DataFrame, parent_path: Path) -> np.ndarray:
    # TODO: load vocabulary from aggregator (LATER BE UPDATED TO RETRIEVE FROM AGGREGATOR'S PUBLIC SITE)
    try:
        shared_file = os.path.join(str(parent_path), AGGREGATOR_DATASITE, "api_data", "netflix_data", "tv-series_vocabulary.json")
        with open(shared_file, "r", encoding="utf-8") as file:
            vocabulary = json.load(file)
    except:
        # TODO: to remove once available in the Aggregator
        with open("./aggregator/data/tv-series_vocabulary.json", "r", encoding="utf-8") as file:
            vocabulary = json.load(file)

    aggregated_data["ID"] = aggregated_data["show"].apply(lambda x: match_title(x, vocabulary))
    
    vector_size = len(vocabulary) 
    sparse_vector = np.zeros(vector_size, dtype=int)

    for _, row in aggregated_data.iterrows():
        if row["ID"] != -1:
            sparse_vector[row["ID"]] += row["Total_Views"]

    unmatched_titles = aggregated_data[aggregated_data["ID"] == -1]["show"].tolist()
    print(">> (create_view_counts_vector) Unmatched Titles:", unmatched_titles)

    return sparse_vector

## ==================================================================================================
## Application functions - Consider refactoring into a separate module
## ==================================================================================================

def check_execution_context(client):
    """
    Determine and handle execution context (aggregator vs. participant).
    """
    if client.email == AGGREGATOR_DATASITE:
        print(f">> {API_NAME} | Running as aggregator.")
        subprocess.run([sys.executable, "aggregator/main.py"])
        sys.exit(0)
    else:
        print(f">> {API_NAME} | Running as participant.")

def setup_environment(client):
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

    def create_public_folder(path: Path, client: Client) -> None:
        """
        Create a API public folder within the specified path.

        This function creates a directory for receiving the private enhanced version \
        of the viewing history.
        """

        os.makedirs(path, exist_ok=True)

        # Set default permissions for this folder
        permissions = SyftPermission.datasite_default(email=client.email)
        permissions.read.append(AGGREGATOR_DATASITE) # set read permission to the aggregator
        permissions.save(path)

    restricted_public_folder = client.api_data(API_NAME)
    create_public_folder(restricted_public_folder, client)
    private_folder = create_private_folder(client.datasite_path, client)
    return restricted_public_folder, private_folder


## ==================================================================================================
## Viewing History Aggregation and Storage
## ==================================================================================================

def aggregate_title_week_counts(reduced_data: np.ndarray) -> np.ndarray:
    """
    Aggregate the reduced viewing history by counting occurrences for each title and week.

    Args:
        reduced_data (np.ndarray): A 2D array with titles and weeks.

    Returns:
        np.ndarray: A 2D array with aggregated counts for each title and week combination.
    """
    counts = Counter(map(tuple, reduced_data))
    aggregated_data = np.array([[title, week, str(count)] for (title, week), count in counts.items()])
    return aggregated_data

def aggregate_and_store_history(reduced_history, viewing_history, private_folder, restricted_public_folder):
    """
    Process and save the reduced, aggregated, and full viewing history.

    Args:
        reduced_history: Reduced viewing history data.
        viewing_history: Full viewing history data.
        private_folder: Path to the private folder.
        restricted_public_folder: Path to the restricted public folder.
    """
    # Aggregate the reduced information
    aggregated_history = aggregate_title_week_counts(reduced_history)

    # Define paths
    public_reduced_file = restricted_public_folder / "netflix_reduced.npy"
    public_aggregated_file = restricted_public_folder / "netflix_aggregated.npy"
    private_full_file = private_folder / "netflix_full.npy"

    # Save reduced viewing history
    np.save(str(public_reduced_file), reduced_history)

    # Save aggregated viewing history
    np.save(str(public_aggregated_file), aggregated_history)

    # Save full viewing history
    np.save(str(private_full_file), viewing_history)

    return aggregated_history

## ==================================================================================================
## Predictor Process
## ==================================================================================================
def train_and_save_mlp(latest_data_file, restricted_public_folder):
    """
    Train the MLP model and save its weights and biases.

    Args:
        latest_data_file: Path to the latest data file.
        restricted_public_folder: Path to the restricted public folder.
    """
    # Train the MLP model
    mlp, _, _, num_samples = train_model(latest_data_file)

    # Define paths
    mlp_weights_file = restricted_public_folder / f"netflix_mlp_weights_{num_samples}.joblib"
    mlp_bias_file = restricted_public_folder / f"netflix_mlp_bias_{num_samples}.joblib"

    # Save MLP weights and biases
    joblib.dump(mlp.coefs_, str(mlp_weights_file))
    joblib.dump(mlp.intercepts_, str(mlp_bias_file))


## ==================================================================================================
## Orchestrator
## ==================================================================================================

def main():

    # Load client
    client = Client.load()

    # Check execution context
    check_execution_context(client)

    # Skip execution if conditions are not met
    if not should_run():
        print(f"Skipping {API_NAME}, not enough time has passed.")
        sys.exit(0)

    # Set up environment
    restricted_public_folder, private_folder = setup_environment(client)

    # Fetch and load Netflix data
    latest_data_file, viewing_history = get_or_download_latest_data()

    # Reduce the original information
    reduced_history = orchestrate_reduction(viewing_history)

    netflix_file_path = 'data/netflix_titles.csv'
    netflix_show_data = load_csv_to_numpy(netflix_file_path)
    my_shows_data = join_viewing_history_with_netflix(reduced_history, netflix_show_data)

    # Next, to perform something useful with this joined (enhanced) data
    # This is an enhanced data compared with the retrieved viewing history from Netflix website
    # show_id,type,title,director,cast,country,date_added,release_year,rating,duration,listed_in,description
    #
    # Useful for Embeddings and more complex learning

    private_shows_file: Path = private_folder / "my_shows_data_full.npy"
    np.save(str(private_shows_file), my_shows_data)

    ##############

    # Process and save watch history
    aggregate_and_store_history(
        reduced_history, 
        viewing_history, 
        private_folder, 
        restricted_public_folder
    )

    # Train and save MLP model
    train_and_save_mlp(latest_data_file, restricted_public_folder)


    # Create a sequence data (filter by > 1 episodes)
    # Columns: series (TV series title), Total_Views (quantity), First_Seen (datetime)
    # - loaded with the original NetflixViewingHistory.csv
    sequence_recommender = SequenceData(viewing_history)
    
    view_counts_vector = create_view_counts_vector(sequence_recommender.aggregated_data, client.datasite_path.parent)
    private_tvseries_views_file: Path = private_folder / "tvseries_views_sparse_vector.npy"
    np.save(str(private_tvseries_views_file), view_counts_vector)
    

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
