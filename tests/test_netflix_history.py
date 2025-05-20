import unittest
from unittest.mock import patch

import numpy as np

from syftbox_netflix.federated_analytics.data_processing import orchestrate_reduction
from syftbox_netflix.participant_utils.data_loading import load_csv_to_numpy


class TestNetflixHistory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Setup class-wide resources before running the tests.
        """
        print("Setting up class resources...")
        cls.test_history = np.array(
            [
                ["The Blacklist: Season 1", "01/01/2023"],
                ["The Blacklist: Season 1", "02/01/2023"],
                ["Movie Title", "02/04/2023"],
                ["Another Movie: Season 4: Chapter 1", "21/10/2023"],
                ["Another Movie: Season 4: Chapter 2", "22/10/2023"],
            ]
        )

    @classmethod
    def tearDownClass(cls):
        """
        Clean up class-wide resources after tests.
        """
        print("Cleaning up class resources...")
        del cls.test_history

    @patch(
        "builtins.open",
        new_callable=unittest.mock.mock_open,
        read_data="Title,Date\nMovie1,01/01/2023\nMovie2,02/01/2023",
    )
    def test_load_netflix_history(self, mock_open):
        file_path = "/fake/path/NetflixViewingHistory.csv"
        result = load_csv_to_numpy(file_path)

        expected = np.array(
            [
                ["Movie1", "01/01/2023"],
                ["Movie2", "02/01/2023"],
            ]
        )
        np.testing.assert_array_equal(result, expected)
        mock_open.assert_called_once_with(file_path, mode="r", encoding="utf-8")

        expected_reduced = np.array(
            [
                ["The Blacklist", "52"],
                ["The Blacklist", "1"],
                ["Movie Title", "13"],
                ["Another Movie", "42"],
                ["Another Movie", "42"],
            ]
        )

        result = orchestrate_reduction(self.test_history)
        np.testing.assert_array_equal(result, expected_reduced)


if __name__ == "__main__":
    unittest.main()
