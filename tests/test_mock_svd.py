import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import numpy as np
import json

# Assume the code provided earlier is in a module named `federated_learning`
from participant.federated_learning.mock_svd import normalize_string, initialize_user_matrix, load_or_initialize_user_matrix, server_initialization

class TestFederatedLearning(unittest.TestCase):

    def test_normalize_string(self):
        # Test the normalize_string function
        self.assertEqual(normalize_string("TestString"), "teststring")
        self.assertEqual(normalize_string("Another String"), "another string")

    @patch("builtins.open", new_callable=mock_open, read_data='{"show1": 0, "show2": 1}')
    @patch("os.path.join", return_value="mock_path")
    @patch("numpy.load")
    def test_server_initialization(self, mock_np_load, mock_path_join, mock_open_file):
        # Mock dependencies for server_initialization
        mock_np_load.return_value = MagicMock()
        mock_np_load.return_value.item.side_effect = lambda: {"show1": 9.0, "show2": 8.5}

        # Run the initialization function
        with patch("os.makedirs") as mock_makedirs:
            server_initialization()

            # Assertions
            mock_makedirs.assert_called()
            mock_np_load.assert_called()

if __name__ == "__main__":
    unittest.main()
