import unittest
import os
import json
import numpy as np
import shutil
from participant.server_utils.data_loading import (
    load_tv_vocabulary,
    load_imdb_ratings,
    load_global_item_factors
)

class TestServerLoader(unittest.TestCase):

    def setUp(self):
        # Setup sandbox environment
        self.sandbox_dir = "test_sandbox/server_loader"
        self.tv_series_path = os.path.join(self.sandbox_dir, "tv_vocabulary.json")
        self.imdb_ratings_path = os.path.join(self.sandbox_dir, "imdb_ratings.npy")
        self.global_v_path = os.path.join(self.sandbox_dir, "global_V.npy")

        os.makedirs(self.sandbox_dir, exist_ok=True)

        # Mock data
        self.tv_vocab = {"show1": 0, "show2": 1}
        self.imdb_ratings = {"Show1": 8.5, "show2": 9.0}
        self.global_v = np.array([[0.21259355, 0.10947686, 0.28493529, 0.70217822],
                          [0.6644103,  0.05377649, 0.6423985,  0.1104549],
                          [0.48359202, 0.42244332, 0.66966578, 0.61169896]])

        # Write mock data
        with open(self.tv_series_path, "w") as f:
            json.dump(self.tv_vocab, f)
        np.save(self.imdb_ratings_path, self.imdb_ratings)
        np.save(self.global_v_path, self.global_v)

    def tearDown(self):
        if os.path.exists(self.sandbox_dir):
            shutil.rmtree(self.sandbox_dir)

    def test_load_tv_vocabulary(self):
        loaded_vocab = load_tv_vocabulary(self.tv_series_path)
        self.assertEqual(loaded_vocab, self.tv_vocab)

    def test_load_imdb_ratings(self):
        loaded_ratings = load_imdb_ratings(self.imdb_ratings_path)
        expected_ratings = {"show1": 8.5, "show2": 9.0} # Normalized keys
        self.assertEqual(loaded_ratings, expected_ratings)

    def test_load_global_item_factors(self):
        loaded_V = load_global_item_factors(self.global_v_path)
        self.assertTrue(np.array_equal(loaded_V, self.global_v))

    def test_missing_files(self):
        with self.assertRaises(FileNotFoundError):
            load_tv_vocabulary("nonexistent.json")
        with self.assertRaises(FileNotFoundError):
            load_imdb_ratings("nonexistent.npy")