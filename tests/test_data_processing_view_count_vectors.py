import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from pathlib import Path
import json
from main import match_title, create_view_counts_vector

class TestDataProcessingViewCountVectors(unittest.TestCase):
    def test_match_title(self):
        """
        Test matching titles for consistency and accuracy.
        """
        vocabulary = {
            "#ABTalks": 0,
            "#NoFilter": 1,
            "Top Gear": 2,
            "South Park": 3,
        }
        # Exact match
        title = "Top Gear"
        result = match_title(title, vocabulary)
        expected = 2
        self.assertEqual(result, expected)

        # Fuzzy match
        title = "SouthPark"
        result = match_title(title, vocabulary, threshold=70)
        expected = 3
        self.assertEqual(result, expected)

        # No match
        title = "Unknown Show"
        result = match_title(title, vocabulary)
        expected = -1
        self.assertEqual(result, expected)

    def test_create_view_counts_vector(self):
        """
        Test creation of view counts vector from viewing history.
        """
        vocabulary = {
            "#ABTalks": 0,
            "#NoFilter": 1,
            "Top Gear": 2,
            "South Park": 3,
        }
        aggregated_data = pd.DataFrame({
            "show": ["Top Gear", "South Park", "Top Gear", "Unknown Show"],
            "Total_Views": [6, 18, 4, 5],
            "First_Seen": ["2012-09-30", "2012-10-21", "2012-09-30", "2013-01-01"],
        })
        parent_path = Path("/mocked/path")

        # Mock the JSON loading
        with patch("builtins.open", unittest.mock.mock_open(read_data=json.dumps(vocabulary))):
            result = create_view_counts_vector(aggregated_data, parent_path)

            expected = np.array([0, 0, 10, 18])  # Top Gear: 6 + 4, South Park: 18
            np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
