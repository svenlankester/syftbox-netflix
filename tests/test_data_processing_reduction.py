import unittest
import numpy as np
from participant.federated_analytics.data_processing import extract_titles, convert_dates_to_weeks, orchestrate_reduction

class TestDataProcessingReduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_history = np.array([
            ["The Blacklist: Season 1", "01/01/2023"],
            ["The Blacklist: Season 1", "02/01/2023"],
            ["Movie Title", "02/04/2023"],
            ["Another Movie: Season 4: Chapter 1", "21/10/2023"],
            ["Another Movie: Season 4: Chapter 2", "22/10/2023"],
        ])

    @classmethod
    def tearDownClass(cls):
        del cls.test_history

    def test_extract_titles(self):
        result = extract_titles(self.test_history)
        expected = np.array([
            "The Blacklist",
            "The Blacklist",
            "Movie Title",
            "Another Movie",
            "Another Movie",
        ])
        np.testing.assert_array_equal(result, expected)

    def test_convert_dates_to_weeks(self):
        result = convert_dates_to_weeks(self.test_history)
        expected = np.array([
            52,  # ISO week of 01/01/2023
            1,   # ISO week of 02/01/2023
            13,  # ISO week of 02/04/2023
            42,  # ISO week of 21/10/2023
            42,  # ISO week of 22/10/2023
        ])
        np.testing.assert_array_equal(result, expected)

    def test_orchestrate_reduction(self):
        reduced = orchestrate_reduction(self.test_history)
        expected = np.array([
            ["The Blacklist", "52"],
            ["The Blacklist", "1"],
            ["Movie Title", "13"],
            ["Another Movie", "42"],
            ["Another Movie", "42"],
        ])
        np.testing.assert_array_equal(reduced, expected)

if __name__ == "__main__":
    unittest.main()