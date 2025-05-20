"""
Test Suite for Aggregator Main Module

This suite includes tests for various functionalities of the aggregator module, ensuring correctness of:
- Network participant discovery
- Parameter extraction and averaging
- Vocabulary creation from TV series data
"""

import json
import shutil
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from syftbox_netflix.aggregator.utils.syftbox import network_participants
from syftbox_netflix.aggregator.utils.vocab import create_tvseries_vocab

API_NAME = "mock_api"
PROJECT_DIR = "test_sandbox"
DATA_DIR = "test_sandbox/aggregator/data"
SHARED_FOLDER = "test_sandbox/this_client/api_data/netflix_data"


class TestAggregatorMain_MLP(unittest.TestCase):
    """
    Test cases for verifying functionalities of aggregator module.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment by creating a sandbox directory.
        """
        cls.base_path = Path(DATA_DIR)
        if cls.base_path.exists():
            shutil.rmtree(cls.base_path)
        cls.base_path.mkdir(parents=True, exist_ok=True)

        cls.shared_path = Path(SHARED_FOLDER)
        if cls.shared_path.exists():
            shutil.rmtree(cls.shared_path)
        cls.shared_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        """
        Clean up the sandbox directory after tests are complete.
        """
        if cls.base_path.exists():
            shutil.rmtree(cls.base_path)

        if cls.shared_path.exists():
            shutil.rmtree(cls.shared_path)

    def setUp(self):
        """
        Ensure a clean state before each test by clearing the sandbox directory.
        """
        for item in self.base_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    def test_network_participants_valid_entries(self):
        """
        Test discovery of network participants with valid API_NAME directories.
        Expected: Only directories containing 'api_data/mock_api' are listed.
        """

        (self.base_path / "user1" / "api_data" / API_NAME).mkdir(parents=True)
        (self.base_path / "user3" / "api_data" / API_NAME).mkdir(parents=True)
        (self.base_path / "user2" / "api_data").mkdir(
            parents=True
        )  # Incomplete structure

        result = network_participants(self.base_path, API_NAME)
        self.assertEqual(result, ["user1", "user3"])

    def test_network_participants_no_valid_entries(self):
        """
        Test behavior when no valid API_NAME directories are present.
        Expected: Empty list.
        """
        (self.base_path / "user1" / "api_data").mkdir(parents=True)
        (self.base_path / "user2" / "api_data").mkdir(parents=True)

        result = network_participants(self.base_path, API_NAME)
        self.assertEqual(result, [])

    def test_network_participants_empty_directory(self):
        """
        Test behavior when the sandbox directory is empty.
        Expected: Empty list.
        """
        result = network_participants(self.base_path, API_NAME)
        self.assertEqual(result, [])

    def test_network_participants_mixed_valid_invalid_entries(self):
        """
        Test discovery of network participants with mixed valid and invalid directories.
        Expected: Only valid directories are listed.
        """
        (self.base_path / "user1" / "api_data" / API_NAME).mkdir(parents=True)
        (self.base_path / "user3" / "api_data" / API_NAME).mkdir(parents=True)
        (self.base_path / "user2" / "api_data").mkdir(
            parents=True
        )  # Incomplete structure
        (self.base_path / "invalid_user").mkdir(parents=True)  # Irrelevant structure

        result = network_participants(self.base_path, API_NAME)
        self.assertEqual(result, ["user1", "user3"])

    @patch("os.getcwd")
    def test_create_tvseries_vocab(self, mock_getcwd):
        """
        Test creation of vocabulary mapping from TV series data.
        Expected: Correct vocabulary mapping JSON file is created from CSV data.
        """
        csv_data = "\n".join(
            [
                "Title",
                "Breaking Bad",
                "Game of Thrones",
                "Breaking Bad",
                "Stranger Things",
            ]
        )

        mock_getcwd.return_value = PROJECT_DIR
        zip_file_path = Path(DATA_DIR) / "netflix_series_2024-12.csv.zip"
        csv_file_path = Path(DATA_DIR) / "netflix_series_2024-12.csv"
        vocab_file_path = Path(SHARED_FOLDER) / "tv-series_vocabulary.json"

        Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

        with open(csv_file_path, "w", encoding="utf-8") as f:
            f.write(csv_data)

        with zipfile.ZipFile(zip_file_path, "w") as zipf:
            zipf.write(csv_file_path, arcname=csv_file_path.name)

        create_tvseries_vocab(SHARED_FOLDER)

        with open(vocab_file_path, "r", encoding="utf-8") as f:
            vocab_mapping = json.load(f)

        expected_vocab = {"Breaking Bad": 0, "Game of Thrones": 1, "Stranger Things": 2}
        self.assertEqual(vocab_mapping, expected_vocab)


if __name__ == "__main__":
    unittest.main()
