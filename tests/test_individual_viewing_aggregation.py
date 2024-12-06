import unittest
import numpy as np
from main import aggregate_title_week_counts, aggregate_and_store_history

class TesIndividiualViewingAggregation(unittest.TestCase):
    def test_aggregate_title_week_counts(self):
        """
        Test aggregation of titles and week counts.
        """
        reduced_data = np.array([
            ["The Blacklist", "52"],
            ["The Blacklist", "1"],
            ["Movie Title", "13"],
            ["Another Movie", "42"],
            ["Another Movie", "42"],
        ])

        result = aggregate_title_week_counts(reduced_data)

        expected = np.array([
            ["The Blacklist", "52", "1"],
            ["The Blacklist", "1", "1"],
            ["Movie Title", "13", "1"],
            ["Another Movie", "42", "2"],
        ])
        np.testing.assert_array_equal(result, expected)

        # Test Case 2: Empty dataset
        reduced_data_empty = np.empty((0, 2))  # Empty array with two columns
        result_empty = aggregate_title_week_counts(reduced_data_empty)
        expected_empty = np.empty(())  # Expected empty array with three columns
        np.testing.assert_array_equal(result_empty, expected_empty)

        # Test Case 3: Case-sensitive titles
        reduced_data_case = np.array([
            ["The Blacklist", "52"],
            ["the blacklist", "1"],
        ])
        result_case = aggregate_title_week_counts(reduced_data_case)
        expected_case = np.array([
            ["The Blacklist", "52", "1"],
            ["the blacklist", "1", "1"],
        ])
        np.testing.assert_array_equal(result_case, expected_case)

        # Test Case 4: Non-sequential or malformed weeks
        reduced_data_weeks = np.array([
            ["Movie Title", "13"],
            ["Movie Title", "9999"],  # Malformed week number
            ["Another Movie", "-42"],  # Negative week number
        ])
        result_weeks = aggregate_title_week_counts(reduced_data_weeks)
        expected_weeks = np.array([
            ["Movie Title", "13", "1"],
            ["Movie Title", "9999", "1"],
            ["Another Movie", "-42", "1"],
        ])
        np.testing.assert_array_equal(result_weeks, expected_weeks)

    def test_aggregate_and_store_history(self):
        """
        TODO - Test aggregation and storage of reduced viewing history.
        """
        return

if __name__ == "__main__":
    unittest.main()
