import re
import joblib
import numpy as np
import os
from pathlib import Path

def get_users_mlp_parameters(
        datasites_path: Path, api_name:str, peers: list[str]
) -> tuple[list, list]:
    """
    This method retrieve the parameters from the local trained MLP. Those files have the names:
    - netflix_mlp_weights_<NUM_SAMPLES>.joblib
    - netflix_mlp_bias_<NUM_SAMPLES>.joblib

    Returns a tuple of (weights, biases) from all participants
    """
    
    weights = []
    biases = []

    for peer in peers:
        dir = datasites_path / peer / "api_data" / api_name

        weight = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and "mlp_weights" in f]
        bias = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and "mlp_bias" in f]
        weight = max(weight, key=extract_number, default=None)  # get the greater 
        bias = max(bias, key=extract_number, default=None)      # get the greater

        try:
            weights.append(dir / weight)
            biases.append(dir / bias)
        except:
            print("There are no participants weights and biases available.")

    return weights, biases

def extract_number(file_name):
    match = re.search(r'_(\d+)\.joblib$', file_name)
    return int(match.group(1)) if match else -1

def weighted_average(parameters, samples):
    total_samples = sum(samples)
    weighted_params = [
        np.multiply(param, n / total_samples) for param, n in zip(parameters, samples)
    ]
    return np.sum(weighted_params, axis=0)

def mlp_fedavg(weights: list, biases: list) -> tuple[list, list]:
    """
    FedAvg computes the weighted average of parameters (weights and biases) from multiple users.
    The weights for averaging are proportional to the number of samples each user has.
    """
    samples = [extract_number(str(n)) for n in weights]
    
    weight_matrices = [joblib.load(weight_path) for weight_path in weights]
    bias_vectors = [joblib.load(bias_path) for bias_path in biases]

    fedavg_weights = [weighted_average([w[layer] for w in weight_matrices], samples) for layer in range(len(weight_matrices[0]))]
    fedavg_biases = [weighted_average([b[layer] for b in bias_vectors], samples) for layer in range(len(bias_vectors[0]))]

    return fedavg_weights, fedavg_biases