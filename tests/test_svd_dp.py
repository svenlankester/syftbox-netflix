import unittest
from unittest.mock import patch, mock_open
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import copy
from participant.federated_learning.svd_dp import (
    apply_differential_privacy,
    plot_delta_distributions,
    calculate_optimal_threshold,
    clip_deltas
)

class TestCalculateOptimalThreshold(unittest.TestCase):

    def setUp(self):
        self.delta_V = {
            0: np.array([0.1, 0.2, 0.3]),
            1: np.array([0.4, 0.5, 0.6]),
            2: np.array([0.7, 0.8, 0.9]),
        }

    def test_mean_threshold(self):
        threshold = calculate_optimal_threshold(self.delta_V, method="mean")
        expected_mean = np.mean([np.linalg.norm(delta) for delta in self.delta_V.values()])
        self.assertAlmostEqual(threshold, expected_mean, places=6)

    def test_median_threshold(self):
        threshold = calculate_optimal_threshold(self.delta_V, method="median")
        expected_median = np.median([np.linalg.norm(delta) for delta in self.delta_V.values()])
        self.assertAlmostEqual(threshold, expected_median, places=6)

    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            calculate_optimal_threshold(self.delta_V, method="invalid")

class TestClipDeltas(unittest.TestCase):

    def setUp(self):
        self.delta_V = {
            0: np.array([0.1, 0.2, 0.3]),
            1: np.array([0.4, 0.5, 0.6]),
            2: np.array([0.7, 0.8, 0.9]),
        }

    def test_manual_clipping_threshold(self):
        clipped_deltas, used_threshold = clip_deltas(self.delta_V.copy(), clipping_threshold=0.5)

        self.assertEqual(used_threshold, 0.5)
        for delta in clipped_deltas.values():
            self.assertLessEqual(np.linalg.norm(delta), 0.5)

    def test_auto_clipping_threshold_median(self):
        clipped_deltas, used_threshold = clip_deltas(self.delta_V.copy(), method="median")

        norms = [np.linalg.norm(delta) for delta in self.delta_V.values()]
        expected_threshold = np.median(norms)

        self.assertAlmostEqual(used_threshold, expected_threshold, places=6)
        for delta in clipped_deltas.values():
            self.assertLessEqual(np.linalg.norm(delta), used_threshold)

class TestApplyDifferentialPrivacy(unittest.TestCase):

    def setUp(self):
        # Sample deltas
        self.delta_V = {
            0: np.array([0.1, 0.2, 0.3]),
            1: np.array([0.4, 0.5, 0.6]),
            2: np.array([0.7, 0.8, 0.9]),
        }
        self.epsilon = 0.5  # Privacy budget
        self.sensitivity = 0.5  # Sensitivity (pre-clipped norm)

    def test_noise_addition(self):
        dp_deltas = apply_differential_privacy(copy.deepcopy(self.delta_V), epsilon=self.epsilon, sensitivity=self.sensitivity)

        for item_id, delta in dp_deltas.items():
            # Ensure the deltas differ significantly
            self.assertFalse(np.allclose(delta, self.delta_V[item_id], atol=1e-6), f"Noise was not added for item {item_id}")

    def test_noise_scale_calculation(self):
        # Test if noise scale is calculated correctly
        with patch("numpy.random.normal") as mock_random_normal:
            mock_random_normal.return_value = np.zeros_like(list(self.delta_V.values())[0])  # Mock noise to avoid randomness
            apply_differential_privacy(copy.deepcopy(self.delta_V), epsilon=self.epsilon, sensitivity=self.sensitivity)

            # Ensure noise scale matches sensitivity / epsilon
            expected_scale = self.sensitivity / self.epsilon
            mock_random_normal.assert_called_with(scale=expected_scale, size=(3,))  # Check if scale matches

    def test_empty_deltas(self):
        # Test with empty delta dictionary
        dp_deltas = apply_differential_privacy({}, epsilon=self.epsilon, sensitivity=self.sensitivity)
        self.assertEqual(dp_deltas, {}, "Function did not handle empty deltas correctly.")

    def test_high_epsilon(self):
        dp_deltas = apply_differential_privacy(copy.deepcopy(self.delta_V), epsilon=1e6, sensitivity=self.sensitivity)

        for item_id, delta in dp_deltas.items():
            # Ensure minimal noise impact
            self.assertTrue(
                np.allclose(delta, self.delta_V[item_id], atol=1e-4),
                "Unexpected noise added for high epsilon."
            )

    def test_low_epsilon(self):
        dp_deltas = apply_differential_privacy(copy.deepcopy(self.delta_V), epsilon=1e-3, sensitivity=self.sensitivity)

        for item_id, delta in dp_deltas.items():
            norm_original = np.linalg.norm(self.delta_V[item_id])
            norm_noised = np.linalg.norm(delta)
            # Ensure the norms differ significantly
            self.assertNotAlmostEqual(norm_original, norm_noised, places=1, msg="Noise not significant for low epsilon.")