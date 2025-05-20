import copy
import unittest
from unittest.mock import patch

import numpy as np

from syftbox_netflix.federated_learning.svd_dp import (
    apply_differential_privacy,
    calculate_optimal_threshold,
    clip_deltas,
    get_noise_function,
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
        expected_mean = np.mean(
            [np.linalg.norm(delta) for delta in self.delta_V.values()]
        )
        self.assertAlmostEqual(threshold, expected_mean, places=6)

    def test_median_threshold(self):
        threshold = calculate_optimal_threshold(self.delta_V, method="median")
        expected_median = np.median(
            [np.linalg.norm(delta) for delta in self.delta_V.values()]
        )
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
        clipped_deltas, used_threshold = clip_deltas(
            self.delta_V.copy(), clipping_threshold=0.5
        )

        self.assertEqual(used_threshold, 0.5)
        for delta in clipped_deltas.values():
            self.assertLessEqual(np.linalg.norm(delta), 0.5)

    def test_auto_clipping_threshold_median(self):
        clipped_deltas, used_threshold = clip_deltas(
            self.delta_V.copy(), method="median"
        )

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
        """Test that noise is added to the deltas."""
        for noise_type in ["gaussian", "laplace"]:
            with self.subTest(noise_type=noise_type):
                dp_deltas = apply_differential_privacy(
                    copy.deepcopy(self.delta_V),
                    epsilon=self.epsilon,
                    sensitivity=self.sensitivity,
                    noise_type=noise_type,
                )
                for item_id, delta in dp_deltas.items():
                    self.assertFalse(
                        np.allclose(delta, self.delta_V[item_id], atol=1e-6),
                        f"Noise was not added for item {item_id} with noise type {noise_type}",
                    )

    def test_noise_scale_calculation(self):
        """Verify noise scale calculation for Gaussian and Laplace noise."""
        for noise_type in ["gaussian", "laplace"]:
            with self.subTest(noise_type=noise_type):
                if noise_type == "gaussian":
                    expected_scale = np.sqrt(2 * np.log(1.25 / 1e-5)) * (
                        self.sensitivity / self.epsilon
                    )
                    patch_target = "numpy.random.normal"
                elif noise_type == "laplace":
                    expected_scale = self.sensitivity / self.epsilon
                    patch_target = "numpy.random.laplace"

                # Mock noise generator with correct shape
                with patch(patch_target) as mock_random:
                    mock_random.return_value = np.zeros(
                        3
                    )  # Correctly shaped mock return
                    apply_differential_privacy(
                        copy.deepcopy(self.delta_V),
                        epsilon=self.epsilon,
                        sensitivity=self.sensitivity,
                        noise_type=noise_type,
                    )
                    # Validate noise function call with expected parameters
                    mock_random.assert_called_with(
                        loc=0, scale=expected_scale, size=(3,)
                    )

    def test_empty_deltas(self):
        """Ensure function handles empty dictionaries correctly."""
        dp_deltas = apply_differential_privacy(
            {}, epsilon=self.epsilon, sensitivity=self.sensitivity
        )
        self.assertEqual(
            dp_deltas, {}, "Function did not handle empty deltas correctly."
        )

    def test_high_epsilon(self):
        """Verify that high epsilon produces minimal noise."""
        dp_deltas = apply_differential_privacy(
            copy.deepcopy(self.delta_V),
            epsilon=1e6,  # High epsilon
            sensitivity=self.sensitivity,
        )
        for item_id, delta in dp_deltas.items():
            self.assertTrue(
                np.allclose(delta, self.delta_V[item_id], atol=1e-4),
                f"Unexpected noise added for item {item_id} with high epsilon.",
            )

    def test_low_epsilon(self):
        """Verify that low epsilon produces significant noise."""
        dp_deltas = apply_differential_privacy(
            copy.deepcopy(self.delta_V),
            epsilon=1e-3,  # Low epsilon
            sensitivity=self.sensitivity,
        )
        for item_id, delta in dp_deltas.items():
            norm_original = np.linalg.norm(self.delta_V[item_id])
            norm_noised = np.linalg.norm(delta)
            self.assertNotAlmostEqual(
                norm_original,
                norm_noised,
                places=1,
                msg=f"Noise not significant for item {item_id} with low epsilon.",
            )

    def test_invalid_noise_type(self):
        """Ensure an invalid noise type raises a ValueError."""
        with self.assertRaises(ValueError):
            apply_differential_privacy(
                copy.deepcopy(self.delta_V),
                epsilon=self.epsilon,
                sensitivity=self.sensitivity,
                noise_type="invalid",
            )

    def test_get_noise_function_gaussian(self):
        """Test that the factory returns a valid Gaussian noise function."""
        noise_func = get_noise_function("gaussian")
        noise = noise_func(self.sensitivity, self.epsilon, size=10)
        self.assertEqual(noise.shape, (10,))
        self.assertIsInstance(noise, np.ndarray)

    def test_get_noise_function_laplace(self):
        """Test that the factory returns a valid Laplace noise function."""
        noise_func = get_noise_function("laplace")
        noise = noise_func(self.sensitivity, self.epsilon, size=10)
        self.assertEqual(noise.shape, (10,))
        self.assertIsInstance(noise, np.ndarray)

    def test_get_noise_function_invalid(self):
        """Ensure invalid noise type raises ValueError."""
        with self.assertRaises(ValueError):
            get_noise_function("invalid")
