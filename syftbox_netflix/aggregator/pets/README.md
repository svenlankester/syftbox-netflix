# Aggregator-Side Federated System

This directory contains aggregator-side components demonstrating two key privacy-preserving strategies: **Federated Analytics** and **Federated Learning**. Both approaches operate without aggregating raw user data centrally, enabling insights and improved models that respect end-user privacy.

## Key Concepts

- **Federated Analytics:**  
  Perform aggregate computations—like finding popular content—across distributed data without collecting individual-level details.
  
- **Federated Learning:**  
  Train a global model by combining locally trained parameters from many users, enhancing the model while keeping personal data on the user’s device.

## Files Overview

### `dp_top5.py` (Federated Analytics)
- **What It Does:**  
  Collects differentially private (DP) vectors of episode-watch counts from multiple participants.
  
- **Goal:**  
  Identify the top-5 most watched series globally without accessing any individual’s raw viewing history.
  
- **How It Works:**  
  1. Participants locally generate DP-protected counts.  
  2. The aggregator combines these anonymized counts.  
  3. The system produces a global ranking of series popularity.
  
- **Benefits:**  
  - Privacy: No direct user-level data exposure.  
  - Insights: Informs decision-making (e.g., content promotion) based on aggregate trends.  
  - Scalability: As more users participate, the popularity measure becomes richer without additional privacy risk.

### `fedavg_mlp.py` (Federated Learning)
- **What It Does:**  
  Implements Federated Averaging (FedAvg) for MLP models. Each user trains an MLP on their local data and shares only model parameters (weights, biases) with the aggregator.
  
- **Goal:**  
  Produce a global model that reflects everyone’s tastes, improving recommendations without centralizing personal data.
  
- **How It Works:**  
  1. Each participant trains an MLP on their device.  
  2. The aggregator collects parameter updates, not raw training data.  
  3. Weighted averaging merges these updates into a single global model.  
  4. The improved global model is shared back, continuously refining recommendations.
  
- **Benefits:**  
  - Enhanced Recommendations: Leverages collective intelligence for personalized, up-to-date suggestions.  
  - Privacy-Preserving: Respects user sovereignty over their data.  
  - Adaptive: Responds to evolving user interests at scale.

## What the Federated System Needs

- **Participants (Clients):**  
  Users running local computations (counting watches, training MLPs) and sharing only privacy-preserving updates.
  
- **Server (Aggregator):**  
  A secure environment that receives anonymized counts or model parameters, performs federated analytics or federated learning steps, and distributes global insights back to clients.

- **Minimal Participant Thresholds:**  
  Ensures that no single user’s data can be inferred. Requires a certain number of participants before producing aggregate outputs.

- **Vocabularies and Metadata:**  
  Mappings (e.g., `tv-series_vocabulary.json`) to interpret aggregated indices or model parameters, enabling meaningful recommendations and insights.


## Next Steps

1. **Scaling to Multiple Users:**  
   Deploy with many participants to fully leverage federated approaches for rich, diverse insights.

2. **Integrating SVD or Hybrid Models:**  
   Combine MLP, SVD, or hybrid recommender models to capture both linear and nonlinear user-item relationships.

3. **Enhanced Privacy Techniques:**  
   Incorporate secure aggregation, differential privacy at all stages, and robust anonymity safeguards.

4. **Automated Continuous Learning:**  
   Implement regular federated training rounds that adapt as user interests evolve, ensuring fresh and dynamic recommendations.
