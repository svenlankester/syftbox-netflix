import unittest

import numpy as np

from syftbox_netflix.federated_analytics.data_processing import (
    calculate_show_ratings,
    join_viewing_history_with_netflix,
)


class TestDataProcessingEnrichment(unittest.TestCase):
    def test_join_viewing_history_with_netflix_normal(self):
        """
        Test normal joining of viewing history with Netflix show data.
        """
        reduced_history = np.array(
            [
                ["The Blacklist", "52"],
                ["Breaking Bad", "12"],
            ]
        )

        netflix_show_data = np.array(
            [
                ["s1", "TV Show", "The Blacklist", "Crime/Drama"],
                ["s2", "TV Show", "Breaking Bad", "Crime/Thriller"],
            ]
        )

        result = join_viewing_history_with_netflix(reduced_history, netflix_show_data)

        expected = np.array(
            [
                [
                    "The Blacklist",
                    "52",
                    "s1",
                    "TV Show",
                    "The Blacklist",
                    "Crime/Drama",
                ],
                [
                    "Breaking Bad",
                    "12",
                    "s2",
                    "TV Show",
                    "Breaking Bad",
                    "Crime/Thriller",
                ],
            ]
        )

        np.testing.assert_array_equal(result, expected)

    def test_join_viewing_history_with_netflix_missing_title(self):
        """
        Test when some titles in viewing history are not in Netflix show data.
        """
        reduced_history = np.array(
            [
                ["The Blacklist", "52"],
                ["Unknown Show", "10"],
            ]
        )

        netflix_show_data = np.array(
            [
                ["s1", "TV Show", "The Blacklist", "Crime/Drama"],
            ]
        )

        result = join_viewing_history_with_netflix(reduced_history, netflix_show_data)

        expected = np.array(
            [
                [
                    "The Blacklist",
                    "52",
                    "s1",
                    "TV Show",
                    "The Blacklist",
                    "Crime/Drama",
                ],
            ]
        )  # "Unknown Show" is not included

        np.testing.assert_array_equal(result, expected)

    def test_join_viewing_history_with_netflix_empty_data(self):
        """
        TODO - Test behavior when either viewing history or Netflix show data is empty.
        """
        return

    def test_join_viewing_history_with_netflix_partial_match(self):
        """
        Test when some titles partially match in Netflix show data.
        """
        reduced_history = np.array(
            [
                ["Blacklist", "52"],
                ["Breaking Bad", "12"],
            ]
        )

        netflix_show_data = np.array(
            [
                ["s1", "TV Show", "The Blacklist", "Crime/Drama"],
                ["s2", "TV Show", "Breaking Bad", "Crime/Thriller"],
            ]
        )

        result = join_viewing_history_with_netflix(reduced_history, netflix_show_data)

        expected = np.array(
            [
                [
                    "Breaking Bad",
                    "12",
                    "s2",
                    "TV Show",
                    "Breaking Bad",
                    "Crime/Thriller",
                ],
            ]
        )  # "Blacklist" does not fully match "The Blacklist"

        np.testing.assert_array_equal(result, expected)

    def test_join_viewing_history_with_netflix_duplicate_titles(self):
        """
        Test behavior when titles in viewing history or Netflix show data are duplicated.
        """
        reduced_history = np.array(
            [
                ["The Blacklist", "52"],
                ["The Blacklist", "53"],
            ]
        )

        netflix_show_data = np.array(
            [
                ["s1", "TV Show", "The Blacklist", "Crime/Drama"],
                ["s2", "TV Show", "The Blacklist", "Crime/Drama"],  # Duplicate entry
            ]
        )

        result = join_viewing_history_with_netflix(reduced_history, netflix_show_data)

        expected = np.array(
            [
                [
                    "The Blacklist",
                    "52",
                    "s2",
                    "TV Show",
                    "The Blacklist",
                    "Crime/Drama",
                ],
                [
                    "The Blacklist",
                    "53",
                    "s2",
                    "TV Show",
                    "The Blacklist",
                    "Crime/Drama",
                ],
            ]
        )  # Last match used, duplicate ignored

        np.testing.assert_array_equal(result, expected)


class TestCalculateShowRatings(unittest.TestCase):
    def test_rule1_multiple_high_weeks(self):
        """
        Test when a show is watched >3 episodes in the same week, for multiple weeks.
        """
        viewing_data = np.array(
            [
                ["Show1", "47", "4"],
                ["Show1", "48", "5"],
            ]
        )
        ratings = calculate_show_ratings(viewing_data)
        self.assertEqual(ratings["Show1"], 5)

    def test_rule2_consecutive_single_episodes(self):
        """
        Test when a show is watched at least 1 episode for three consecutive weeks.
        """
        viewing_data = np.array(
            [
                ["Show2", "47", "1"],
                ["Show2", "48", "2"],
                ["Show2", "49", "1"],
            ]
        )
        ratings = calculate_show_ratings(viewing_data)
        self.assertEqual(ratings["Show2"], 5)

    def test_rule3_single_high_week(self):
        """
        Test when a show is watched >4 episodes in a single week.
        """
        viewing_data = np.array(
            [
                ["Show3", "47", "5"],
            ]
        )
        ratings = calculate_show_ratings(viewing_data)
        self.assertEqual(ratings["Show3"], 4)

    def test_rule4_multiple_weeks_with_high_views(self):
        """
        Test when a show is watched >4 episodes across multiple weeks.
        """
        viewing_data = np.array(
            [
                ["Show4", "47", "2"],
                ["Show4", "48", "3"],  # Total = 5, across multiple weeks
            ]
        )
        ratings = calculate_show_ratings(viewing_data)
        self.assertEqual(ratings["Show4"], 3)

    def test_rule5_total_views_greater_than_three(self):
        """
        Test when a show has >3 episodes watched in total.
        """
        viewing_data = np.array(
            [
                ["Show5", "47", "2"],
                ["Show5", "48", "2"],  # Total = 4
            ]
        )
        ratings = calculate_show_ratings(viewing_data)
        self.assertEqual(ratings["Show5"], 2)

    def test_rule6_default_low_views(self):
        """
        Test when a show does not meet any of the above conditions.
        """
        viewing_data = np.array(
            [
                ["Show6", "47", "1"],
            ]
        )
        ratings = calculate_show_ratings(viewing_data)
        self.assertEqual(ratings["Show6"], 1)

    def test_empty_data(self):
        """
        Test behavior when the input data is empty.
        """
        viewing_data = np.empty((0, 3))
        ratings = calculate_show_ratings(viewing_data)
        self.assertEqual(ratings, {})

    def test_multiple_shows_with_mixed_conditions(self):
        """
        Test when multiple shows meet different conditions.
        """
        viewing_data = np.array(
            [
                ["Show7", "47", "5"],  # Rule 3: 4 stars
                ["Show8", "47", "1"],  # Rule 6: 1 star
                ["Show8", "48", "1"],  # Rule 2: 5 stars
                ["Show8", "49", "1"],  # Rule 2: 5 stars
                ["Show9", "47", "4"],  # Rule 1: 5 stars
                ["Show9", "48", "4"],  # Rule 1: 5 stars
            ]
        )
        ratings = calculate_show_ratings(viewing_data)
        self.assertEqual(ratings["Show7"], 4)
        self.assertEqual(ratings["Show8"], 5)
        self.assertEqual(ratings["Show9"], 5)


if __name__ == "__main__":
    unittest.main()
