# Federated Recommendation System (SVD Mock Example)

This repository demonstrates a **mock federated recommendation system** using a single user’s local updates and global model aggregation. Although we focus on a single user for demonstration, the intended scenario is where multiple clients (users) train locally and share updates with a central server to continuously refine a global model.

The relevant file is: `participant/federated_learning/mock_svd.py`

## Overview

The code simulates:

1. **Local Training with SVD (Matrix Factorization)**:  
   We apply a simplified form of SVD-based collaborative filtering. Given a user-item rating matrix, SVD decomposes it into user and item latent factors. Here, we create a user and item embedding space and iteratively update these factors based on the user’s local ratings.

2. **Federated Learning-Like Updates**:  
   While there’s only one user in this mock setup, the code mimics how a user’s device would:
   - Train/update latent factors locally.
   - Compute a delta (the difference in updated item factors).
   - Send these deltas back to a “server” that applies them to the global model parameters.

3. **Local Recommendations Based on Recency**:  
   The system generates recommendations by focusing on the user’s most recently watched shows. We compute a “recent profile” vector from the embeddings of recently consumed items and use it to score candidate shows.

4. **Dynamic Item Introduction**:  
   If the user selects an item that is not in the vocabulary, we dynamically add it, initialize its latent factors, and incorporate the feedback into the global model.

## Theory & Model Choices

### Collaborative Filtering and SVD

**Collaborative filtering** relies on learning from patterns across many users. By factorizing the user-item rating matrix (via SVD or related methods), we discover latent factors representing abstract concepts (e.g., user preferences for certain genres or item attributes).

- **SVD Strengths**:  
  - Simple, effective for large-scale systems.
  - Provides straightforward latent factor representations for users and items.
  
- **SVD Weaknesses**:  
  - Limited in capturing complex nonlinear relationships.
  - Requires multiple users to find meaningful latent factors. With one user, the factorization is not very informative.
  
### MLP-Based Methods

**Neural network-based approaches (MLP-based)** replace the simple dot product of latent factors with nonlinear layers. They can learn more complex interactions and incorporate side information (e.g., genres, metadata) easily.

- **MLP Strengths**:  
  - Can model nonlinear relationships.
  - Flexible integration of additional features.
  
- **MLP Weaknesses**:  
  - More complex and potentially harder to train.
  - More computationally intensive.
  - Requires larger datasets to avoid overfitting.

In a federated setting with multiple users, MLP-based recommenders could leverage rich item/user metadata and interactions. SVD is a simpler starting point, easier to implement and understand. MLP-based approaches are a natural next step once the basic federated training pipeline is established and validated.

### Current Choice

We start with **SVD-based CF** for simplicity. Once the federated pipeline works with SVD, we could experiment with MLP approaches to potentially achieve better recommendation quality, especially when the user base and item catalog are large and diverse.

## Key Steps in the Code

1. **Training a Local Model**:  
   We simulate training by using gradient updates to refine user and item embeddings.

2. **Persistence**:  
   After local training, we save the global model parameters (`global_U.npy` and `global_V.npy`) to disk, representing the server’s global state.

3. **Local Recommendations**:  
   We compute recommendations by focusing on recent viewing history. Using the updated latent factors, we predict and rank items to recommend.

4. **User Feedback and Model Updating**:  
   If the user chooses an item not in the top recommendations, we treat that as feedback. The local model updates its parameters, computes a delta, and “sends” it to the server. The server then integrates these updates, continuously improving the model.

## Next Steps

Besides having this work properly between aggregator and participant app, we will:

1. **Multiple Users Simulation**:  
   Introduce multiple simulated users, each with their own local data. Aggregate their updates on the server to observe more meaningful collaborative patterns.

2. **Federated Averaging Protocol**:  
   Implement a proper federated averaging (FedAvg) step, where multiple user updates are aggregated (e.g., by averaging gradients or factor changes) on the server. There is a version already available in the aggregator app, so it may be matter of just integrating it.

3. **Privacy Enhancements**:  
   Explore differential privacy or secure aggregation methods to ensure the user’s data is not revealed through model updates.

4. **Extend to MLP**:  
   After validating the federated pipeline with SVD, consider moving to an MLP-based recommendation model. Incorporate item metadata, user demographics (if available), or content embeddings to capture richer preference signals.

