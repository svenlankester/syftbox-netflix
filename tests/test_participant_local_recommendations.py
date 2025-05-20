import os
import unittest

import numpy as np

from syftbox_netflix.federated_learning.mock_svd import local_recommendation


class TestParticipantLocalRecommendation(unittest.TestCase):
    def setUp(self):
        self.sandbox_dir = "test_sandbox/participant_datasite/local_recommendation"
        self.user_id = "test_user"
        self.shared_path = os.path.join(self.sandbox_dir, "shared")
        self.restricted_path = os.path.join(
            self.sandbox_dir, "restricted", self.user_id
        )
        self.private_folder = os.path.join(self.sandbox_dir, "private", self.user_id)

        # Create sandbox directories
        os.makedirs(self.shared_path, exist_ok=True)
        os.makedirs(os.path.join(self.private_folder, "svd_training"), exist_ok=True)
        os.makedirs(os.path.join(self.restricted_path, "svd_training"), exist_ok=True)

        # Declare input/output paths
        self.global_V_path = os.path.join(self.shared_path, "global_V.npy")
        self.participant_V_path = os.path.join(
            self.private_folder, "svd_training", "updated_V.npy"
        )
        self.user_matrix_path = os.path.join(
            self.private_folder, "svd_training", "U.npy"
        )
        self.delta_V_path = os.path.join(
            self.restricted_path, "svd_training", "delta_V.npy"
        )

        self.user_ratings = [
            ("show1", 12, 5, 4.5),  # Watched in week 12
            ("show2", 11, 3, 3.0),  # Watched in week 11
        ]
        self.tv_vocab = {"show1": 0, "show2": 1, "show3": 2, "show4": 3}
        self.global_V = np.random.normal(size=(4, 10))  # 3 items, latent dimension 10
        self.user_U = np.random.normal(size=(10,))

        # Save mock data
        np.save(self.global_V_path, self.global_V)
        np.save(self.user_matrix_path, self.user_U)

    def test_local_recommendation(self):
        # Call the local recommendation function
        recommendations = local_recommendation(
            local_path=self.private_folder,
            global_path=self.shared_path,
            tv_vocab=self.tv_vocab,
            user_ratings=self.user_ratings,
            exclude_watched=True,  # Test with watched shows excluded
        )

        # Validate recommendations
        self.assertEqual(
            len(recommendations), 2
        )  # Only non-watched items should be recommended

        # Validate top recommendations are sorted by predicted rating
        self.assertGreaterEqual(recommendations[0][2], recommendations[1][2])

        # Check for a specific item if needed
        self.assertIn(
            "show3", [rec[0] for rec in recommendations]
        )  # Ensure "show3" is recommended
        self.assertIn(
            "show4", [rec[0] for rec in recommendations]
        )  # Ensure "show3" is recommended

    def test_no_recent_items(self):
        user_ratings = [
            ("show1", 10, 5, 4.5),  # Watched long ago
        ]
        recommendations = local_recommendation(
            local_path=self.private_folder,
            global_path=self.shared_path,
            tv_vocab=self.tv_vocab,
            user_ratings=user_ratings,
            exclude_watched=True,
        )
        self.assertGreater(len(recommendations), 0)  # Should still recommend something

    def test_all_items_watched(self):
        user_ratings = [
            ("show1", 12, 5, 4.5),
            ("show2", 12, 3, 3.0),
            ("show3", 12, 2, 2.5),  # All shows watched
            ("show4", 12, 1, 1.0),
        ]
        recommendations = local_recommendation(
            local_path=self.private_folder,
            global_path=self.shared_path,
            tv_vocab=self.tv_vocab,
            user_ratings=user_ratings,
            exclude_watched=True,
        )
        self.assertEqual(len(recommendations), 0)  # No recommendations possible

    def test_missing_global_data(self):
        os.remove(self.global_V_path)  # Simulate missing global data
        with self.assertRaises(FileNotFoundError):
            local_recommendation(
                local_path=self.private_folder,
                global_path=self.shared_path,
                tv_vocab=self.tv_vocab,
                user_ratings=self.user_ratings,
                exclude_watched=True,
            )
