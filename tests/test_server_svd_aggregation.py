import os
import shutil
import unittest
from unittest import mock

import numpy as np

from syftbox_netflix.federated_learning.mock_svd import server_aggregate
from syftbox_netflix.federated_learning.svd_server_aggregation import (
    add_differential_privacy_noise,
    aggregate_item_factors,
    calculate_aggregated_delta,
    clip_updates,
    normalize_weights,
    validate_weights,
)


class TestServerAggregate(unittest.TestCase):
    def setUp(self):
        """Set up sandbox environment and mock data."""
        self.sandbox_dir = "test_sandbox/server_aggregate"
        self.save_path = os.path.join(self.sandbox_dir, "tmp_model_parms")
        self.global_V_path = os.path.join(self.save_path, "global_V.npy")

        os.makedirs(self.save_path, exist_ok=True)

        self.V = np.random.rand(3, 4)  # Global item factors
        self.updates = [
            {0: np.array([0.1, 0.2, 0.3, 0.2]), 1: np.array([0.4, 0.5, 0.6, 0.9])},
            {1: np.array([0.7, 0.8, 0.9, 0.6]), 2: np.array([0.2, 0.3, 0.4, 0.1])},
        ]

        np.save(self.global_V_path, self.V)

    def tearDown(self):
        """Clean up the sandbox environment."""
        if os.path.exists(self.sandbox_dir):
            shutil.rmtree(self.sandbox_dir)

    def test_server_aggregate(self):
        """Test the full server aggregation process."""
        server_aggregate(self.updates, weights=[0.3, 0.7], save_to=self.save_path)
        updated_V = np.load(self.global_V_path)
        self.assertEqual(updated_V.shape, self.V.shape)


class TestAggregateItemFactors(unittest.TestCase):
    def setUp(self):
        """Set up mock data for testing."""
        # Set seed
        np.random.seed(42)
        self.V = np.array(
            [[0.4, 1, 0.8, 0.6], [0.7, 0.8, 0.8, 1.5], [0.7, 0.9, 0.3, 1]]
        )

        np.random.rand(3, 4)  # Global item factors (3 items, latent dim 4)
        self.updates = [
            {0: np.array([0.1, 0.2, 0.3, 0.2]), 1: np.array([0.4, 0.5, 0.6, 0.9])},
            {1: np.array([0.7, 0.8, 0.9, 0.6]), 2: np.array([0.2, 0.3, 0.4, 0.1])},
        ]

    def test_aggregation_with_weighted_updates_sum_to_one(self):
        """Test aggregation with weighted updates."""
        weights = [0.3, 0.7]
        updated_V = aggregate_item_factors(
            self.V,
            self.updates,
            weights=weights,
            learning_rate=1.0,
            clipping_threshold=None,
            epsilon=None,
        )

        expected_V = np.array(
            [
                [0.43, 1.06, 0.89, 0.66],
                [1.31, 1.51, 1.61, 2.19],
                [0.84, 1.11, 0.58, 1.07],
            ]
        )

        # Test equality of updated_V and expected_V numpy arrays
        np.testing.assert_almost_equal(updated_V, expected_V, decimal=6)

    def test_dp_noise_added(self):
        """Test that differential privacy noise is added."""
        epsilon = 1.0
        clipping_threshold = 1.0
        updated_V = aggregate_item_factors(
            self.V, self.updates, epsilon=epsilon, clipping_threshold=clipping_threshold
        )
        self.assertFalse(np.allclose(updated_V, self.V, atol=1e-4))


class TestNormalizeWeights(unittest.TestCase):
    def test_valid_weights(self):
        """Test valid weights."""
        weights = [0.3, 0.4, 0.3]
        try:
            validate_weights(weights, num_participants=3)
        except ValueError:
            self.fail("validate_weights raised ValueError unexpectedly!")

    def test_mismatched_weights(self):
        """Test mismatched number of weights."""
        with self.assertRaises(ValueError):
            validate_weights([0.3, 0.4], num_participants=3)

    def test_zero_total_weight(self):
        """Test zero total weight."""
        with self.assertRaises(ValueError):
            validate_weights([0, 0, 0], num_participants=3)

    def test_none_weights(self):
        """Test None weights (no validation performed)."""
        try:
            validate_weights(None, num_participants=3)
        except ValueError:
            self.fail(
                "validate_weights raised ValueError unexpectedly for None weights!"
            )

    def test_normal_weights(self):
        """Test normalization of a list of positive weights."""
        weights = [0.3, 0.4, 0.3]
        normalized = normalize_weights(weights, num_participants=3)
        self.assertAlmostEqual(sum(normalized), 1.0, places=6)
        self.assertEqual(len(normalized), len(weights))
        self.assertListEqual(normalized, [0.3, 0.4, 0.3])  # Already normalized

    def test_unequal_weights(self):
        """Test normalization of a list of unequal weights."""
        weights = [1, 2, 3]
        normalized = normalize_weights(weights, num_participants=3)
        self.assertAlmostEqual(sum(normalized), 1.0, places=6)
        expected = [w / sum(weights) for w in weights]
        self.assertListEqual(normalized, expected)

    def test_zero_weights(self):
        """Test that a ValueError is raised when total weight is zero."""
        with self.assertRaises(ValueError):
            normalize_weights([0, 0, 0], num_participants=3)


class TestClipUpdates(unittest.TestCase):
    def setUp(self):
        """Set up mock updates for testing."""
        self.updates = [
            {0: np.array([0.5, 0.5]), 1: np.array([1.0, 1.0])},
            {0: np.array([0.3, 0.4]), 1: np.array([2.0, 2.0])},
        ]

    def test_clipping_applied(self):
        """Test that updates are clipped to the threshold."""
        clipping_threshold = 1.0
        clipped = clip_updates(self.updates, clipping_threshold)

        # Validate clipping for each item
        for delta_V in clipped:
            for delta in delta_V.values():
                self.assertLessEqual(np.linalg.norm(delta), clipping_threshold)

    def test_no_clipping_needed(self):
        """Test that updates are unchanged when below the threshold."""
        clipping_threshold = 3.0
        clipped = clip_updates(self.updates, clipping_threshold)
        for original, clipped_delta in zip(self.updates, clipped):
            for item_id in original:
                np.testing.assert_array_equal(original[item_id], clipped_delta[item_id])

    def test_empty_updates(self):
        """Test that an empty update list returns an empty list."""
        clipped = clip_updates([], clipping_threshold=1.0)
        self.assertEqual(clipped, [])


class TestAddDifferentialPrivacyNoise(unittest.TestCase):
    def setUp(self):
        """Set up mock aggregated deltas for testing."""
        self.aggregated_delta = {
            0: np.array([0.5, 0.5]),
            1: np.array([1.0, 1.0]),
            2: np.array([0.3, 0.4]),
        }

    def test_noise_addition(self):
        """Test that noise is added to the deltas."""
        epsilon = 1.0
        clipping_threshold = 1.0
        noised_delta = add_differential_privacy_noise(
            aggregated_delta=self.aggregated_delta,
            epsilon=epsilon,
            clipping_threshold=clipping_threshold,
        )

        # Check that values differ from the original
        for item_id, original_delta in self.aggregated_delta.items():
            self.assertFalse(
                np.allclose(noised_delta[item_id], original_delta, atol=1e-6)
            )

    def test_noise_scale(self):
        """Test that noise scale is calculated correctly."""
        epsilon = 1.0
        clipping_threshold = 1.0
        noise_scale = clipping_threshold / epsilon

        # Patch np.random.normal to validate its arguments
        with mock.patch("numpy.random.normal") as mock_normal:
            mock_normal.return_value = np.zeros(2)  # Dummy noise
            add_differential_privacy_noise(
                aggregated_delta=self.aggregated_delta,
                epsilon=epsilon,
                clipping_threshold=clipping_threshold,
            )
            # Ensure np.random.normal is called with correct scale
            mock_normal.assert_any_call(scale=noise_scale, size=(2,))

    def test_empty_aggregated_delta(self):
        """Test that an empty aggregated delta returns an empty result."""
        noised_delta = add_differential_privacy_noise(
            {}, epsilon=1.0, clipping_threshold=1.0
        )
        self.assertEqual(noised_delta, {})


class TestCalculateAggregatedDelta(unittest.TestCase):
    def setUp(self):
        """Set up mock data for testing."""
        self.V = np.random.rand(3, 4)  # Global item factors (3 items, latent dim 4)
        self.updates = [
            {0: np.array([0.1, 0.2, 0.3, 0.4]), 1: np.array([0.5, 0.6, 0.7, 0.8])},
            {1: np.array([0.2, 0.3, 0.4, 0.5]), 2: np.array([0.3, 0.4, 0.5, 0.6])},
        ]
        self.normalized_weights = [0.4, 0.6]
        self.learning_rate = 1.0

    def test_valid_updates(self):
        """Test valid updates with normalized weights."""
        aggregated_delta = calculate_aggregated_delta(
            self.V, self.updates, self.normalized_weights, self.learning_rate
        )

        # Validate that aggregated_delta has the correct structure
        self.assertEqual(len(aggregated_delta), len(self.V))
        for item_id in aggregated_delta:
            self.assertTrue(item_id in [0, 1, 2])  # Ensure all items are accounted for

    def test_empty_updates(self):
        """Test that empty updates return zero deltas."""
        updates = []
        aggregated_delta = calculate_aggregated_delta(
            self.V, updates, self.normalized_weights, self.learning_rate
        )

        # Check that all aggregated deltas are zero
        for delta in aggregated_delta.values():
            self.assertTrue(np.all(delta == 0))

    def test_zero_weights(self):
        """Test that zero weights result in zero deltas."""
        weights = [0.0, 0.0]
        aggregated_delta = calculate_aggregated_delta(
            self.V, self.updates, weights, self.learning_rate
        )

        # Check that all aggregated deltas are zero
        for delta in aggregated_delta.values():
            self.assertTrue(np.all(delta == 0))

    def test_learning_rate_scaling(self):
        """Test that learning rate scales the aggregated deltas."""
        learning_rate = 2.0
        aggregated_delta = calculate_aggregated_delta(
            self.V, self.updates, self.normalized_weights, learning_rate
        )

        # Validate that deltas are scaled by the learning rate
        expected_aggregated_delta = {
            item_id: np.zeros_like(self.V[0]) for item_id in range(len(self.V))
        }
        for i, delta_V in enumerate(self.updates):
            for item_id, delta in delta_V.items():
                expected_aggregated_delta[item_id] += (
                    delta * self.normalized_weights[i] * learning_rate
                )

        for item_id in expected_aggregated_delta:
            np.testing.assert_array_almost_equal(
                aggregated_delta[item_id], expected_aggregated_delta[item_id], decimal=6
            )
