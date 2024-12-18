import os
import sys
import numpy as np
from pathlib import Path
from syftbox.lib import Client
from participant_utils.checks import should_run
from participant_utils.syftbox import setup_environment
from participant_utils.data_loading import load_csv_to_numpy, get_or_download_latest_data

# Package functions
from loaders.netflix_loader import participants_datasets
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
NETFLIX_PROFILE = os.getenv("NETFLIX_PROFILE", None)
NETFLIX_PROFILES = os.getenv("NETFLIX_PROFILES", NETFLIX_PROFILE)


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

    title_genre_dict = fa.create_title_field_dict(netflix_show_data, title_col=2, field_col=10) # tmp dict - may be useful for aggregates.
    user_information = fa.add_column_from_dict(aggregated_history, ratings_dict, key_col=0, new_col_name='rating')

    # This is an enhanced data compared with the retrieved viewing history from Netflix website
    # Useful for more complex analytics
    my_shows_data = fa.join_viewing_history_with_netflix(user_information, netflix_show_data)

    # Save data
    fa.save_npy_data(restricted_public_folder, "netflix_reduced.npy", reduced_history)
    fa.save_npy_data(restricted_public_folder, "netflix_aggregated.npy", user_information)
    fa.save_npy_data(private_folder, "netflix_full.npy", viewing_history)
    fa.save_npy_data(private_folder, "data_full.npy", my_shows_data)
    fa.save_npy_data(private_folder, "ratings.npy", ratings_dict)

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

    for profile in NETFLIX_PROFILES.split(","):
        if NETFLIX_PROFILE:
            # Tmp not to break the existing code and references
            restricted_public_folder, private_folder = setup_environment(client, API_NAME, AGGREGATOR_DATASITE)
        else:
            restricted_public_folder, private_folder = setup_environment(client, API_NAME, AGGREGATOR_DATASITE, profile)

        # Try to retrieve user data from datasets.yaml
        dataset_yaml = participants_datasets(client.datasite_path, dataset_name = "Netflix Data", dataset_format = "CSV")
        
        if (dataset_yaml is None):
            # if not available on datasets.yaml, Fetch and load Netflix data 
            latest_data_file, viewing_history = get_or_download_latest_data(OUTPUT_DIR, CSV_NAME, profile)
        else:
            print(f">> Retrieving data from datasets.yaml: {dataset_yaml}")
            latest_data_file = dataset_yaml
            try: 
                viewing_history = load_csv_to_numpy(dataset_yaml)
            except Exception as e:
                print(f"[Error] to load retrieved path for NetflixViewingHistory.csv from datasets.yaml \n{e}")
                sys.exit(1)

        # Run private processes and write to public/private/restricted directories
        run_federated_analytics(restricted_public_folder, private_folder, viewing_history)
        # run_federated_learning(AGGREGATOR_DATASITE, restricted_public_folder, private_folder, viewing_history, latest_data_file, client.datasite_path.parent)
        # run_top5_dp(private_folder / "tvseries_views_sparse_vector.npy", restricted_public_folder, verbose=False)
        ##############

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        sys.exit(1)


