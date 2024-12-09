import os
import json
import numpy as np

from participant.main import setup_environment
from syftbox.lib import Client

from dotenv import load_dotenv
load_dotenv()
API_NAME = os.getenv("API_NAME")
AGGREGATOR_DATASITE = os.getenv("AGGREGATOR_DATASITE")

client = Client.load()
restricted_public_folder, private_folder = setup_environment(client, API_NAME, AGGREGATOR_DATASITE)

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
    normalized_ratings = {normalize_string(title): (float(rating) / 2 if rating else 0) for title, rating in imdb_data.items()}

    # Step 3: Initialize item factors V
    k = 10  # Latent dimensionality
    num_items = max(tv_vocab.values()) + 1  # Ensure V covers all items in the vocabulary
    np.random.seed(42)  # For reproducibility

    not_found = 0
    default_rating = np.mean(list(normalized_ratings.values()))
    V = np.random.normal(scale=0.01, size=(num_items, k))  # Random initialization
    for title, idx in tv_vocab.items():
        normalized_title = normalize_string(title)
        if normalized_title in normalized_ratings:
            V[idx] *= normalized_ratings[normalized_title]  # Scale by normalized IMDB ratings
        else:
            V[idx] *= default_rating  # Assign default rating
            not_found += 1

    print(f"Initialized item factors for {len(V)} shows. {not_found} shows not found in IMDB data.")

    # Step 4: Save the initialized model
    save_to = "mock_dataset_location/tmp_model_parms"
    os.makedirs(save_to, exist_ok=True)
    np.save(os.path.join(save_to, "global_V.npy"), V)

    print("Server initialization complete. Item factors (V) are saved.")

def initialize_user_matrix(user_id, latent_dim, save_path="mock_dataset_location/tmp_model_parms"):
    import numpy as np
    import os

    # Create save directory if not exists
    os.makedirs(save_path, exist_ok=True)

    # Initialize user matrix
    U_u = np.random.normal(scale=0.01, size=(latent_dim,))

    # Save user matrix
    user_matrix_path = os.path.join(save_path, f"{user_id}_U.npy")
    np.save(user_matrix_path, U_u)
    print(f"Initialized and saved user matrix for {user_id}.")
    return U_u


def load_or_initialize_user_matrix(user_id, latent_dim, save_path="mock_dataset_location/tmp_model_parms"):
    import os
    import numpy as np

    user_matrix_path = os.path.join(save_path, f"{user_id}_U.npy")
    if os.path.exists(user_matrix_path):
        U_u = np.load(user_matrix_path)
        print(f"Loaded existing user matrix for {user_id}.")
    else:
        U_u = initialize_user_matrix(user_id, latent_dim, save_path)
    return U_u


def participant_fine_tuning(user_id, private_folder):

    # Step 1: Load vocabulary and global item factors V
    with open("aggregator/data/tv-series_vocabulary.json", "r") as f:
        tv_vocab = json.load(f)

    global_V_path = os.path.join("mock_dataset_location/tmp_model_parms", "global_V.npy")
    V = np.load(global_V_path)

    # Step 4: Initialize user factors
    k = V.shape[1]  # Latent dimensionality

    # Step 2: Load participant's ratings
    my_ratings_path = os.path.join(private_folder, f'{user_id}_ratings.npy')
    final_ratings = np.load(my_ratings_path, allow_pickle=True).item()

    # Load or initialize user matrix
    U_u = load_or_initialize_user_matrix(user_id, k, save_path="mock_dataset_location/tmp_model_parms")

    # Map titles to item IDs
    item_ids = {title: tv_vocab[title] for title in final_ratings if title in tv_vocab}

    # Step 3: Construct training data for the participant
    train_data = [(user_id, item_ids[t], final_ratings[t]) for t in final_ratings if t in item_ids]

    # Step 5: Local training
    alpha = 0.01  # Learning rate
    lambda_reg = 0.1
    iterations = 10  # Number of training iterations

    for it in range(iterations):
        for (_, item_id, r) in train_data:
            # Predict
            pred = U_u.dot(V[item_id])
            error = r - pred

            # Gradient updates
            U_u_grad = error * V[item_id] - lambda_reg * U_u
            V_i_grad = error * U_u - lambda_reg * V[item_id]

            U_u += alpha * U_u_grad
            V[item_id] += alpha * V_i_grad

    # Step 6: Send updates to the server
    delta_V = {item_id: V[item_id] for (_, item_id, _) in train_data}  # Changes to item factors
    print(f"Participant {user_id} finished training and updated item factors.")

    # Save updated V locally for debugging or future updates
    participant_save_path = os.path.join("mock_dataset_location/tmp_model_parms", f"{user_id}_updated_V.npy")
    np.save(participant_save_path, V)

    # Step 5: Persist updated user matrix
    user_matrix_path = os.path.join("mock_dataset_location/tmp_model_parms", f"{user_id}_U.npy")
    np.save(user_matrix_path, U_u)
    print(f"Updated and saved user matrix for {user_id}.")

    return delta_V  # This delta can be sent to the server

def server_aggregate(updates):
    import numpy as np
    import os

    # Step 1: Load current global V
    global_V_path = os.path.join("mock_dataset_location/tmp_model_parms", "global_V.npy")
    V = np.load(global_V_path)

    # Step 2: Aggregate updates
    for delta_V in updates:
        for item_id, delta in delta_V.items():
            V[item_id] += delta

    # Step 3: Save the updated global V
    np.save(global_V_path, V)
    print("Server aggregation complete. Global item factors (V) updated.")


########################################
# Step 0: Model Initialisation and Fine-Tuning
########################################

# Server initialisation
server_initialization()

# Participant fine-tuning
user1_id = "myshows"
user2_id = "mysister" #sister

delta_V_1 = participant_fine_tuning(user1_id, private_folder)
delta_V_2 = participant_fine_tuning(user2_id, private_folder)
# Check deltas are different
assert delta_V_1 != delta_V_2

# Server aggregation
# server_aggregate([delta_V_1, delta_V_2])


########################################
# Step 1: Local Recommendation Computation
########################################

def local_recommendation(user_id, tv_vocab, user_ratings):
    # Assume we have user_ratings, global_V, global_U, tv_vocab, etc. from previous code

    # Load model parameters
    global_path = "mock_dataset_location/tmp_model_parms"
    local_path = "mock_dataset_location/tmp_model_parms"
    global_V_path = os.path.join(global_path, "global_V.npy")
    user_U_path = os.path.join(local_path, f"{user_id}_U.npy")

    user_U = np.load(user_U_path)
    global_V = np.load(global_V_path)

    print("Selecting recommendations based on most recent shows watched...")
    recent_week = 47
    recent_items = [title for (title, week, rating) in user_ratings if week == recent_week]
    recent_item_ids = [tv_vocab[title] for title in recent_items if title in tv_vocab]
    print("For week (of all years)", recent_week, "recently watched n_shows=:", len(recent_items))

    if recent_item_ids:
        U_recent = sum(global_V[item_id] for item_id in recent_item_ids) / len(recent_item_ids)
    else:
        U_recent = user_U  # fallback

    all_items = list(tv_vocab.keys())
    watched_titles = set(t for (t, _, _) in user_ratings)

    # Optionally, exclude already watched items
    # candidate_items = [title for title in all_items if title not in watched_titles]

    # Or consider all items
    candidate_items = all_items

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
    return top_6

with open("aggregator/data/tv-series_vocabulary.json", "r") as f:
    tv_vocab = json.load(f)


# Example user data
my_activity_path = os.path.join(restricted_public_folder, 'netflix_aggregated.npy')
my_activity = np.load(my_activity_path, allow_pickle=True) # Title, Week, Rating

my_activity_formatted = np.empty(my_activity.shape, dtype=object)
my_activity_formatted[:, 0] = my_activity[:, 0]  # Show name remains as string
my_activity_formatted[:, 1] = my_activity[:, 1].astype(int)  # Week number as int
my_activity_formatted[:, 2] = my_activity[:, 2].astype(int)  # View times as int

top_6 = local_recommendation(user1_id, tv_vocab, user_ratings=my_activity_formatted)

print("Recalculating recommendations to verify consistency...")
top_6 = local_recommendation(user1_id, tv_vocab, user_ratings=my_activity_formatted)



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
new_rating = 5
new_row = np.array([user_new_choice, 47, new_rating], dtype=object)
my_activity_formatted = np.vstack([my_activity_formatted, new_row])

print("Recalculating recommendations after user interaction to verify consistency...")
top_6 = local_recommendation(user1_id, tv_vocab, user_ratings=my_activity_formatted)

# We now have an additional data point. The user updates their model locally.

########################################
# Step 3: Local Update (Incremental Training)
########################################

# Load model parameters
load_from = "mock_dataset_location/tmp_model_parms"
global_U_path = os.path.join(load_from, "global_U.npy")
global_V_path = os.path.join(load_from, "global_V.npy")

global_U = np.load(global_U_path)
global_V = np.load(global_V_path)

# Identify item_id for the newly chosen item
# If user_new_choice not in tv_vocab, add it dynamically:
if user_new_choice not in tv_vocab:
    print(f"Item '{user_new_choice}' not in vocabulary. Adding it now.")

    # Find a new item_id for this show
    new_item_id = max(tv_vocab.values()) + 1
    tv_vocab[user_new_choice] = new_item_id

    # Initialize item factors randomly
    k = global_U.shape[0]  # latent dimension (assuming global_U is [k])
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
pred_before = global_U.dot(global_V[new_item_id])
error = new_rating - pred_before

# Compute gradients
U_u_grad = error * global_V[new_item_id] - lambda_reg * global_U
V_i_grad = error * global_U - lambda_reg * global_V[new_item_id]

# Store the old item factors to compute delta
old_V_item = global_V[new_item_id].copy()

# Update locally
global_U += alpha * U_u_grad
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
server_global_U = np.load(os.path.join(save_to, "global_U.npy"))

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
np.save(os.path.join(save_to, "global_U.npy"), server_global_U)
np.save(os.path.join(save_to, "global_V.npy"), server_global_V)

print("\nServer: Applied client delta updates to global parameters and re-saved.")

# At this point, the server’s global model now reflects the user’s latest interaction
# with the newly chosen item, and this item is also integrated into the vocabulary.

### Re-run local recommendation to see if the new item is now recommended
print("Recalculating recommendations after user interaction and model update to verify consistency...")
local_recommendation(tv_vocab, user_ratings=my_activity_formatted, recalculate_global=False)



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
