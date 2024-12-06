import unittest
from unittest.mock import patch, MagicMock
import os
from datetime import datetime
from main import (
    download_daily_data,
    get_latest_file,
    get_or_download_latest_data,
)

class TestNetflixLoader(unittest.TestCase):
    @patch("main.NetflixFetcher")
    @patch("os.path.exists")
    def test_download_daily_data(self, mock_exists, mock_fetcher):
        mock_fetcher_instance = MagicMock()
        mock_fetcher.return_value = mock_fetcher_instance

        mock_exists.return_value = True

        output_dir = "/mocked/output"
        file_name = "NetflixViewingActivity.csv"
        file_path = os.path.join(output_dir, file_name)

        download_daily_data(output_dir, file_name)

        mock_fetcher.assert_called_once_with(output_dir)
        mock_fetcher_instance.run.assert_called_once()
        mock_exists.assert_called_once_with(file_path)

    @patch("main.CSV_NAME", "NetflixViewingActivity.csv")
    @patch("os.listdir", return_value=[
        "NetflixViewingActivity_2024-11-27.csv",
        "NetflixViewingActivity_2024-11-26.csv",
        "NetflixViewingActivity_2024-11-25.csv",
    ])
    @patch("os.path.isfile", return_value=True)
    def test_get_latest_file(self, mock_isfile, mock_listdir):
        subfolder_path = "/mocked/subfolder"

        expected_file = "/mocked/subfolder/NetflixViewingActivity_2024-11-27.csv"
        result = get_latest_file(subfolder_path)

        self.assertEqual(result, expected_file)
        mock_listdir.assert_called_once_with(subfolder_path)

    @patch("main.get_latest_file")
    @patch("main.download_daily_data")
    @patch("os.path.exists")
    @patch("main.OUTPUT_DIR", "/mocked/netflix_data")
    @patch("main.CSV_NAME", "NetflixViewingActivity.csv")
    @patch("main.datetime")
    @patch("main.load_csv_to_numpy")
    def test_get_or_download_latest_data_file_exists(self, mock_load_csv_to_numpy, mock_datetime, mock_exists, mock_download, mock_get_latest):
        """
        Test when today's file already exists.
        """
        mock_datetime.now.return_value = datetime(2024, 11, 27)
        mock_datetime.strftime = datetime.strftime

        today_date = "2024-11-27"
        file_path = f"/mocked/netflix_data/NetflixViewingActivity_{today_date}.csv"

        mock_exists.return_value = True
        mock_get_latest.return_value = file_path

        result = get_or_download_latest_data()

        mock_download.assert_not_called()
        mock_get_latest.assert_called_once_with("/mocked/netflix_data")
        mock_load_csv_to_numpy.assert_called_once_with(file_path)

    @patch("main.get_latest_file")
    @patch("main.download_daily_data")
    @patch("os.path.exists")
    @patch("main.OUTPUT_DIR", "/mocked/netflix_data")
    @patch("main.CSV_NAME", "NetflixViewingActivity.csv")
    @patch("main.datetime")
    @patch("main.load_csv_to_numpy")
    def test_get_or_download_latest_data_file_missing(self, mock_load_csv_to_numpy, mock_datetime, mock_exists, mock_download, mock_get_latest):
        """
        Test when today's file does not exist, and download is triggered.
        """
        mock_datetime.now.return_value = datetime(2024, 11, 27)
        mock_datetime.strftime = datetime.strftime

        today_date = "2024-11-27"
        file_path = f"/mocked/netflix_data/NetflixViewingActivity_{today_date}.csv"

        mock_exists.return_value = False
        mock_get_latest.return_value = file_path

        result = get_or_download_latest_data()

        mock_download.assert_called_once()
        mock_get_latest.assert_called_once_with("/mocked/netflix_data")
        mock_load_csv_to_numpy.assert_called_once_with(file_path)

if __name__ == "__main__":
    unittest.main()
