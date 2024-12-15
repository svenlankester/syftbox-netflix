import os
import json
import numpy as np
import shutil
import diffprivlib.tools as dp
import matplotlib.pyplot as plt
import copy

from participant.main import setup_environment
from syftbox.lib import Client


from participant.federated_learning.svd_participant_finetuning import participant_fine_tuning

from dotenv import load_dotenv
load_dotenv()


def normalize_string(s):
    """
    """
    return s.replace('\u200b', '').lower()

def server_initialization():
    import numpy as np
    import os
    import json

    # Step 1: Load vocabulary
    with open("aggregator/data/tv-series_vocabulary.json", "r") as f:
        tv_vocab = json.load(f)

    # Step 2: Load and normalize IMDB ratings
    imdb_ratings_path = os.path.join("data", 'imdb_ratings.npy')
    imdb_data = np.load(imdb_ratings_path, allow_pickle=True).item()
    imdb_ratings = {normalize_string(title): float(rating) for title, rating in imdb_data.items() if rating}

    # Step 3: Initialize item factors V
    k = 10  # Latent dimensionality
    num_items = max(tv_vocab.values()) + 1  # Ensure V covers all items in the vocabulary
    np.random.seed(42)  # For reproducibility


    not_found = 0
    default_rating = np.mean(list(imdb_ratings.values()))  # Mean IMDB rating [1, 10]
    k = 10  # Latent dimensionality
    num_items = max(tv_vocab.values()) + 1  # Total number of items

    # Initialize V directly based on IMDB ratings or default rating
    V = np.zeros((num_items, k))

    for title, idx in tv_vocab.items():
        normalized_title = normalize_string(title)
        
        if normalized_title in imdb_ratings:
            rating = imdb_ratings[normalized_title]
        else:
            # rating = np.random.normal(loc=default_rating, scale=0.5)  # Random value close to default_rating
            rating = np.random.normal(loc=default_rating, scale=0.1, size=k)

            not_found += 1

        # Create a base latent representation proportional to the IMDB rating
        # Higher ratings should lead to stronger initial vectors
        base_vector = np.full(k, rating)

        # Add variability across dimensions by adding small random noise around the base rating
        V[idx] = base_vector + np.random.normal(scale=0.2 * rating, size=k)

    # Normalize each item's vector to have unit norm (ensuring diverse yet consistent scales)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)

    print(f"Initialized item factors for {len(V)} shows. {not_found} shows not found in IMDB data.")

    # Step 4: Save the initialized model
    save_to = "mock_dataset_location/tmp_model_parms"
    os.makedirs(save_to, exist_ok=True)
    np.save(os.path.join(save_to, "global_V.npy"), V)

    print("Server initialization complete. Item factors (V) are saved.")

def server_aggregate(updates, weights=None, learning_rate=1.0, epsilon=4.0, clipping_threshold=0.5):
    """
    Aggregate updates from participants using a weighted average.
    Missing updates for specific items are handled gracefully.

    Args:
        updates (list[dict]): List of delta dictionaries from participants.
                              Each dictionary maps item_id -> delta vector.
        weights (list[float]): List of weights for each participant. If None, equal weight is assumed.
        learning_rate (float): Scaling factor for the aggregated deltas.
    """
    import numpy as np
    import os

    # Step 1: Load current global V
    global_V_path = os.path.join("mock_dataset_location/tmp_model_parms", "global_V.npy")
    V = np.load(global_V_path)

    # Step 2: Initialize weights
    if weights is None:
        weights = [1.0] * len(updates)  # Equal weight for all participants

    if len(weights) != len(updates):
        raise ValueError("The number of weights must match the number of updates.")

    # Normalize weights to sum to 1
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Step 3: Aggregate updates with clipping and DP noise
    aggregated_delta = {item_id: np.zeros_like(V[item_id]) for item_id in range(len(V))}


    for i, delta_V in enumerate(updates):
        weight = normalized_weights[i] * learning_rate
        for item_id, delta in delta_V.items():
            # Clip updates
            if clipping_threshold:
                norm = np.linalg.norm(delta)
                if norm > clipping_threshold:
                    delta = (delta / norm) * clipping_threshold
            # Weighted aggregation
            aggregated_delta[item_id] += weight * delta

    # Add DP noise to the aggregated updates
    # if epsilon:
    #     noise_scale = clipping_threshold / epsilon
    #     for item_id in aggregated_delta:
    #         noise = np.random.normal(scale=noise_scale, size=aggregated_delta[item_id].shape)
    #         aggregated_delta[item_id] += noise

    if epsilon:
        if clipping_threshold is None:
            clipping_threshold = np.max([np.linalg.norm(delta) for delta in aggregated_delta.values()])
        noise_scale = clipping_threshold / epsilon
        for item_id in delta_V:
            delta_norm = np.linalg.norm(delta_V[item_id])
            if delta_norm > 0:  # Avoid division by zero
                # Normalize, add noise, and rescale
                aggregated_delta[item_id] /= delta_norm
                noise = np.random.normal(scale=noise_scale, size=aggregated_delta[item_id].shape)
                aggregated_delta[item_id] += noise
                aggregated_delta[item_id] *= delta_norm

    else:
        # Normal aggregation
        for i, delta_V in enumerate(updates):
            weight = normalized_weights[i] * learning_rate
            for item_id, delta in delta_V.items():
                V[item_id] += weight * delta  # Apply weighted delta for this participant

    # Step 4: Apply the aggregated updates to the global model
    for item_id, delta in aggregated_delta.items():
        V[item_id] += delta
    


    # Step 4: Save the updated global V
    np.save(global_V_path, V)
    print("Server aggregation complete. Global item factors (V) updated.")

def local_recommendation(user_id, tv_vocab, user_ratings, exclude_watched=True):
    # Assume we have user_ratings, global_V, global_U, tv_vocab, etc. from previous code

    # Load model parameters
    global_path = "mock_dataset_location/tmp_model_parms"
    local_path = os.path.join("mock_dataset_location/tmp_model_parms", user_id)
    global_V_path = os.path.join(global_path, "global_V.npy")
    user_U_path = os.path.join(local_path, f"{user_id}_U.npy")

    user_U = np.load(user_U_path)
    global_V = np.load(global_V_path)

    print("Selecting recommendations based on most recent shows watched...")
    recent_week = 12
    recent_items = [title for (title, week, n_watched, rating) in user_ratings if week == recent_week]
    recent_item_ids = [tv_vocab[title] for title in recent_items if title in tv_vocab]
    print("For week (of all years)", recent_week, "watched n_shows=:", len(recent_items))

    alpha = 0.7  # Weight for long-term preferences
    beta = 0.3   # Weight for recent preferences

    if recent_item_ids:
        U_global_activity = sum(global_V[item_id] for item_id in recent_item_ids) / len(recent_item_ids)
        U_recent = alpha * user_U + beta * U_global_activity
    else:
        U_recent = user_U  # fallback

    all_items = list(tv_vocab.keys())
    watched_titles = set(normalize_string(t) for (t, _, _, _) in user_ratings)

    # Optionally, exclude already watched items
    if exclude_watched:
        candidate_items = [title for title in all_items if normalize_string(title) not in watched_titles]
    else:
        # Or consider all items
        candidate_items = all_items


    # tv_vocab = {normalize_string(title): item_id for title, item_id in tv_vocab.items()}

    predictions = []
    for title in candidate_items:
        item_id = tv_vocab[title]
        pred_rating = U_recent.dot(global_V[item_id])
        predictions.append((title, pred_rating))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_6 = predictions[:6]

    print("Recommended based on most recently watched:")
    for i, (show, score) in enumerate(top_6):
        print(f"\t{i+1} => {show}: {score:.4f}")


    # Debug for development...
    # # Analytics for the shows that are rated by user
    # print("User's ratings:")
    # ratings = set([(title, rating) for title, _, _, rating in user_ratings if title in tv_vocab])

    # print("Actual Ratings for Recommended Shows:")
    # for title,rating in ratings:
    #     if title in [x[0] for x in top_6]:
    #         print(f"\t{title}: {rating}")
    
    # print("Actual Ratings for recently watched shows:")
    # for title,rating in ratings:
    #     if title in recent_items:
    #         print(f"\t{title}: {rating}")

    return top_6

def run_process():

    API_NAME = os.getenv("API_NAME")
    AGGREGATOR_DATASITE = os.getenv("AGGREGATOR_DATASITE")
    NETFLIX_PROFILES = os.getenv("NETFLIX_PROFILES")

    client = Client.load()

    restricted_public_folders = {}
    private_folders = {}

    for profile in NETFLIX_PROFILES.split(","):
        restricted_public_folders[profile], private_folders[profile] = setup_environment(client, API_NAME, AGGREGATOR_DATASITE, profile)

    ########################################
    # Step 0: Model Initialisation and Fine-Tuning
    ########################################

    # Clear folder
    fldr_base = "mock_dataset_location/tmp_model_parms"
    user_ids = ["myshows", "mysister", "mydad", "mymom", "mykid"]
    user_ids = ["myshows", "mydad", "mymom", "mykid"]

    for user_id in user_ids:
        fldr_user = os.path.join(fldr_base, user_id)
        if os.path.exists(fldr_user):
            shutil.rmtree(fldr_user)

    # Server initialisation
    server_initialization()
    backup_global_v = np.load("mock_dataset_location/tmp_model_parms/global_V.npy") # For analytics

    delta_V = {}
    for user_id in user_ids:
        # Fine-tuning of the item embeddings with user data
        delta_V[user_id] = participant_fine_tuning(user_id, private_folders[user_id], epsilon=10, clipping_threshold=None) #0.36

    ########################################
    # Step 1: Local Recommendation Computation
    ########################################

    with open("aggregator/data/tv-series_vocabulary.json", "r") as f:
        tv_vocab = json.load(f)

    # Example user data
    my_activity_path = os.path.join(restricted_public_folders['myshows'], 'netflix_aggregated.npy')
    my_activity = np.load(my_activity_path, allow_pickle=True) # Title, Week, Rating

    my_activity_formatted = np.empty(my_activity.shape, dtype=object)
    my_activity_formatted[:, 0] = my_activity[:, 0]  # Show name remains as string
    my_activity_formatted[:, 1] = my_activity[:, 1].astype(int)  # Week number as int
    my_activity_formatted[:, 2] = my_activity[:, 2].astype(int)  # View times as int
    my_activity_formatted[:, 3] = my_activity[:, 3].astype(float)  # Ratings as float

    print("Vanilla Recommendations (IMDB)...")
    top_6 = local_recommendation('myshows', tv_vocab, user_ratings=my_activity_formatted)

    print("Updating Global Model with user deltas...")
    # Server aggregation
    # server_aggregate([delta_V['myshows'], delta_V['mysister']])
    delta_V_list = list(delta_V.values())
    server_aggregate(delta_V_list, epsilon=10, clipping_threshold=None)

    print("Federated Recommendations (IMDB)...")
    top_6 = local_recommendation('myshows', tv_vocab, user_ratings=my_activity_formatted)


    # Logs
    global_V_path = os.path.join("mock_dataset_location/tmp_model_parms", "global_V.npy")
    global_V = np.load(global_V_path)
    print("Global V shape:", global_V.shape)

    top_show = top_6[0][0]
    top_show_id = tv_vocab[top_show]
    print(f"Top show '{top_show}' has item_id={top_show_id}")

    # # Debug: Check the actual rating for the top show
    # for user in user_ids:
    #     user_rating_path = os.path.join(private_folders[user], 'ratings.npy')
    #     user_rating = np.load(user_rating_path, allow_pickle=True).item()
    #     print(f"Actual rating for '{top_show}' by {user}: {user_rating.get(top_show, "Not rated")}")


    import pandas as pd
    # Prepare the top 6 show IDs
    top_shows = [show[0] for show in top_6]
    ratings_matrix = []

    for user in user_ids:
        # Path to the user's ratings
        user_rating_path = os.path.join(private_folders[user], 'ratings.npy')
        
        # Load the user's ratings
        user_rating = np.load(user_rating_path, allow_pickle=True).item()
        
        # Collect ratings for the top 6 shows
        user_ratings = [float(user_rating.get(show_id, np.nan)) for show_id in top_shows]
        ratings_matrix.append(user_ratings)

    # Add imdb rating
    imdb_path = os.path.join("data", 'imdb_ratings.npy')
    imdb_data = np.load(imdb_path, allow_pickle=True).item()
    imdb_data = {title: float(rating)/2 if rating else np.nan for title, rating in imdb_data.items() if rating}

    imdb_ratings = [imdb_data.get(show, np.nan) for show in top_shows]
    ratings_matrix.append(imdb_ratings)

    average_ratings = np.nanmean(ratings_matrix, axis=0)
    ratings_matrix.append(average_ratings)

    # Convert to a DataFrame for better visualization
    ratings_df = pd.DataFrame(ratings_matrix, index=user_ids + ['imdb', 'average'], columns=[show[0] for show in top_6])
    print(ratings_df)


    # Embeddings before and after fine-tuning
    print("Top Show V factors before fine-tuning:")
    print(backup_global_v[top_show_id])

    print("Top Show V factors after fine-tuning:")
    print(global_V[top_show_id])

    pass

    ########################################
    # NOTE -> BELOW HERE IS IN DEVELOPMENT!!!
    ########################################




    ########################################
    # Step 2: User Chooses Something Outside Our Predictions
    ########################################

    # Let's say the user picks a show not in top_5, e.g., "Pedro Páramo" is re-watched or a new title "100 Humans".
    # For demonstration:
    user_new_choice = top_6[-1][0]
    if user_new_choice not in [t for (t, _) in top_6[:5]]:
        print(f"\nUser selected '{user_new_choice}' which was not in the top 5 predictions.")

    # Let's assume the user watched and implicitly "rated" it.
    # Row to append
    new_rating = 3.6
    new_row = np.array([user_new_choice, 47, 1, new_rating], dtype=object)
    my_activity_formatted = np.vstack([my_activity_formatted, new_row])

    print(f"---->Mock user activity updated with the new show={user_new_choice} and rating={new_rating}.")

    # print("Recalculating recommendations after user interaction to verify consistency...")
    # top_6 = local_recommendation(user1_id, tv_vocab, user_ratings=my_activity_formatted)

    # We now have an additional data point. The user updates their model locally.

    ########################################
    # Step 3: Local Update (Incremental Training)
    ########################################

    # Load model parameters
    load_from = "mock_dataset_location/tmp_model_parms"
    local_U_path = os.path.join(load_from, "myshows", "myshows_U.npy")
    global_V_path = os.path.join(load_from, "global_V.npy")

    local_U = np.load(local_U_path)
    global_V = np.load(global_V_path)

    # Identify item_id for the newly chosen item
    # If user_new_choice not in tv_vocab, add it dynamically:
    if user_new_choice not in tv_vocab:
        print(f"Item '{user_new_choice}' not in vocabulary. Adding it now.")

        # Find a new item_id for this show
        new_item_id = max(tv_vocab.values()) + 1
        tv_vocab[user_new_choice] = new_item_id

        # Initialize item factors randomly
        k = local_U.shape[0]  # latent dimension (assuming global_U is [k])
        new_item_factors = np.random.normal(scale=0.01, size=(k,))
        # Expand global_V to accommodate this new item
        # Assuming global_V is shape [num_items, k]
        # We'll need to append a new row
        global_V = np.vstack([global_V, new_item_factors[np.newaxis, :]])
        
        print(f"Assigned item_id={new_item_id} for new show '{user_new_choice}'")
    else:
        new_item_id = tv_vocab[user_new_choice]

    # Now we have new_item_id for the chosen show.
    # Perform a mini step of gradient descent to incorporate the new rating
    alpha = 0.01
    lambda_reg = 0.1

    # Current prediction before update
    pred_before = local_U.dot(global_V[new_item_id])
    error = new_rating - pred_before

    # Compute gradients
    U_u_grad = error * global_V[new_item_id] - lambda_reg * local_U
    V_i_grad = error * local_U - lambda_reg * global_V[new_item_id]

    # Store the old item factors to compute delta
    old_V_item = global_V[new_item_id].copy()

    # Update locally
    local_U += alpha * U_u_grad
    global_V[new_item_id] += alpha * V_i_grad

    # Compute the delta for the item factor
    delta_V = global_V[new_item_id] - old_V_item

    ########################################
    # Step 4: Send Updates (Delta) Back to Server
    ########################################

    # In a real federated scenario, we wouldn't send the raw delta as is,
    # we might send gradient updates or encrypted parameters.
    # For demonstration, let's just print what would be sent.
    print("\nSending updates back to the server:")
    print(f"Item factor delta for item_id {new_item_id} ({user_new_choice}): {delta_V}")

    # Server-side pseudo-code to handle updates:
    # In reality, the server would:
    # - Load the currently stored global_V
    # - Apply the delta to the corresponding item_id

    # Mock server aggregation:
    save_to = "mock_dataset_location/tmp_model_parms"
    os.makedirs(save_to, exist_ok=True)

    # Mock server load:
    server_global_V = np.load(os.path.join(save_to, "global_V.npy"))
    server_local_U = np.load(os.path.join(save_to, "myshows", "myshows_U.npy"))

    # If we added a new item not previously in server_global_V, we need to align the dimensions.
    # Assume server_global_V shape: [N_items, k]
    # If new_item_id >= server_global_V.shape[0], we must expand server_global_V as well.
    if new_item_id >= server_global_V.shape[0]:
        # Expand server_global_V to accommodate new_item_id
        rows_to_add = new_item_id - server_global_V.shape[0] + 1
        additional_rows = np.random.normal(scale=0.01, size=(rows_to_add, server_global_V.shape[1]))
        server_global_V = np.vstack([server_global_V, additional_rows])

    # Apply delta:
    server_global_V[new_item_id] += delta_V

    # Save updated global parameters:
    np.save(os.path.join(save_to, "myshows", "myshows_U.npy"), server_local_U)
    np.save(os.path.join(save_to, "global_V.npy"), server_global_V)

    print("\nServer: Applied client delta updates to global parameters and re-saved.")

    # At this point, the server’s global model now reflects the user’s latest interaction
    # with the newly chosen item, and this item is also integrated into the vocabulary.

    ### Re-run local recommendation to see if the new item is now recommended
    print("Recalculating recommendations after user interaction and model update to verify consistency...")
    local_recommendation("myshows", tv_vocab, user_ratings=my_activity_formatted[:-1])


if __name__ == "__main__":
    run_process()

########################################
# Notes:
########################################
# - This code is conceptual. In a real federated learning framework:
#   - The user factors (global_U here) might be kept entirely local and not shared at all.
#   - Only item factors or gradient updates would be shared, and potentially in a privacy-preserving manner.
# - If multiple users exist, the server collects such deltas from all users and aggregates them.
#   E.g., server_global_V = average of all user updates for each item.

# - If the user picks something unexpected, it affects the local model.
#   Over time, these adjustments help the global model better reflect real user preferences. 
