import unittest

import numpy as np

from syftbox_netflix.federated_learning.mock_svd import normalize_string
from syftbox_netflix.federated_learning.svd_server_initialisation import (
    generate_item_vector,
    get_rating_with_fallback,
    initialize_item_factors,
    normalize_vectors,
)


class TestNormalizeString(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_string("Hello\u200bWorld"), "helloworld")
        self.assertEqual(normalize_string("TEST"), "test")
        self.assertEqual(normalize_string(" "), " ")

    def test_empty_string(self):
        self.assertEqual(normalize_string(""), "")

    def test_no_changes_needed(self):
        self.assertEqual(normalize_string("hello"), "hello")


class TestInitializeItemFactors(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.tv_vocab = {
            "Breaking Bad": 0,
            "Game of Thrones": 1,
            "Unknown Show": 2,
        }
        self.imdb_ratings = {
            "breaking bad": 9.5,
            "game of thrones": 9.3,
        }
        self.latent_dim = 10
        self.random_seed = 42

    def test_initialization_with_all_titles_found(self):
        """Test when all TV vocabulary titles are found in IMDB ratings."""
        tv_vocab = {"Breaking Bad": 0, "Game of Thrones": 1}
        imdb_ratings = {"breaking bad": 9.5, "game of thrones": 9.3}

        V = initialize_item_factors(
            tv_vocab,
            imdb_ratings,
            latent_dim=self.latent_dim,
            random_seed=self.random_seed,
        )

        self.assertEqual(V.shape, (2, self.latent_dim))
        np.testing.assert_allclose(
            np.linalg.norm(V, axis=1), 1, err_msg="Row vectors are not unit norm"
        )

    def test_initialization_with_missing_titles(self):
        """Test behavior when some titles are missing in IMDB ratings."""
        V = initialize_item_factors(
            self.tv_vocab,
            self.imdb_ratings,
            latent_dim=self.latent_dim,
            random_seed=self.random_seed,
        )

        self.assertEqual(V.shape, (3, self.latent_dim))
        np.testing.assert_allclose(
            np.linalg.norm(V, axis=1), 1, err_msg="Row vectors are not unit norm"
        )

    def test_random_seed_reproducibility(self):
        """Test reproducibility with a fixed random seed."""
        V1 = initialize_item_factors(
            self.tv_vocab, self.imdb_ratings, latent_dim=self.latent_dim, random_seed=42
        )
        V2 = initialize_item_factors(
            self.tv_vocab, self.imdb_ratings, latent_dim=self.latent_dim, random_seed=42
        )

        np.testing.assert_array_almost_equal(
            V1, V2, err_msg="Reproducibility failed with the same random seed"
        )

    def test_normalized_vectors(self):
        """Test that all row vectors are normalized."""
        V = initialize_item_factors(
            self.tv_vocab,
            self.imdb_ratings,
            latent_dim=self.latent_dim,
            random_seed=self.random_seed,
        )

        norms = np.linalg.norm(V, axis=1)
        np.testing.assert_allclose(
            norms, np.ones_like(norms), err_msg="Vectors are not normalized"
        )

    def test_large_vocab(self):
        """Test performance with a large vocabulary."""
        tv_vocab = {f"Show {i}": i for i in range(1000)}
        imdb_ratings = {f"show {i}": np.random.uniform(1, 10) for i in range(500)}

        V = initialize_item_factors(
            tv_vocab,
            imdb_ratings,
            latent_dim=self.latent_dim,
            random_seed=self.random_seed,
        )

        self.assertEqual(V.shape, (1000, self.latent_dim))
        np.testing.assert_allclose(
            np.linalg.norm(V, axis=1),
            1,
            err_msg="Vectors are not normalized for large vocab",
        )

    def test_get_rating_with_fallback_found(self):
        """Test retrieving a rating when the title exists."""
        rating = get_rating_with_fallback(
            "breaking bad", self.imdb_ratings, default_rating=7.0
        )
        self.assertEqual(rating, 9.5)

    def test_get_rating_with_fallback_missing(self):
        """Test using the default rating when the title is missing."""
        rating = get_rating_with_fallback(
            "unknown show", self.imdb_ratings, default_rating=7.0
        )
        self.assertEqual(rating, 7.0)

    def test_generate_item_vector(self):
        """Test generating an item vector with noise."""
        rating = 9.0
        vector = generate_item_vector(rating, self.latent_dim)

        # Ensure vector has correct shape
        self.assertEqual(vector.shape, (self.latent_dim,))

        # Check that the vector values are within a reasonable range
        expected_min = rating - 0.2 * rating  # Minimum with added noise
        expected_max = rating + 0.2 * rating  # Maximum with added noise
        self.assertTrue(
            np.all((vector >= expected_min) & (vector <= expected_max)),
            f"Vector values are out of expected range: {vector}",
        )

    def test_normalize_vectors(self):
        """Test normalizing vectors to unit length."""
        V = np.array([[3, 4], [0, 0], [1, 2]])
        normalized = normalize_vectors(V)
        np.testing.assert_allclose(
            np.linalg.norm(normalized, axis=1), [1, 0, 1], atol=1e-6
        )

    # def test_initialization(self):
    #     # start to think of end-to-end testing
    #     # Run server initialization
    #     server_initialization(save_to=self.save_path, tv_series_path=self.vocab_path, imdb_ratings_path=self.imdb_ratings_path)

    #     # Check if the global_V file was created
    #     self.assertTrue(os.path.isfile(self.global_V_path))

    #     # Validate the saved data
    #     global_V = np.load(self.global_V_path)
    #     self.assertEqual(global_V.shape[0], 2)  # Two items in the vocabulary
    #     self.assertEqual(global_V.shape[1], 10)  # Latent dimension


if __name__ == "__main__":
    unittest.main()
