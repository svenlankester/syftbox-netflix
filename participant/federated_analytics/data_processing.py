import numpy as np
from datetime import datetime
from collections import Counter, defaultdict

## ==================================================================================================
## Data Processing (1) - Reduction
## ==================================================================================================

def extract_titles(history: np.ndarray) -> np.ndarray:
    """
    Extract and reduce titles from the viewing history.
    """
    return np.array([title.split(":")[0] if ":" in title else title for title in history[:, 0]])

def convert_dates_to_weeks(history: np.ndarray) -> np.ndarray:
    """
    Convert viewing dates to ISO week numbers.
    """
    return np.array([
        datetime.strptime(date, "%d/%m/%Y").isocalendar()[1]
        for date in history[:, 1]
    ])

def orchestrate_reduction(history: np.ndarray) -> np.ndarray:
    """
    Orchestrates the reduction process for Netflix viewing history.
    """
    titles = extract_titles(history)
    weeks = convert_dates_to_weeks(history)
    return np.column_stack((titles, weeks))

## ==================================================================================================
## Data Processing (2) - Data Enrichment (shows)
## ==================================================================================================

def create_title_genre_dict(imdb_data, title_col=1, genre_col=7):
    """
    Create a dictionary mapping titles to genres from IMDb data.
    Currently not in use as found genres within the Netflix data.

    Args:
        imdb_data (np.ndarray): The IMDb data array.
        title_col (int): The column index for titles in the IMDb data.
        genre_col (int): The column index for genres in the IMDb data.

    Returns:
        dict: A dictionary mapping titles to genres.
    """
    # Create a dictionary with title as the key and genre as the value
    title_genre_dict = {
        str(row[title_col]): str(row[genre_col]) for row in imdb_data if len(row) > max(title_col, genre_col)
    }
    return title_genre_dict

def add_column_from_dict(data, lookup_dict, key_col, new_col_name="new_column"):
    """
    Add a new column to the data based on a lookup dictionary.

    Args:
        data (np.ndarray): The input data array.
        lookup_dict (dict): A dictionary where keys correspond to data[key_col] values
                            and values are the new column entries to add.
        key_col (int): The column index used to match keys in the lookup dictionary.
        new_col_name (str): The name of the new column (optional, for logging/debugging).

    Returns:
        np.ndarray: The augmented data array with the new column added.
    """
    # Add the new column based on the lookup dictionary
    new_column = [lookup_dict.get(row[key_col], "Unknown") for row in data]
    augmented_data = np.column_stack((data, new_column))
    print(f"Added column '{new_col_name}' to the data.")
    return augmented_data

def join_viewing_history_with_netflix(reduced_history, netflix_show_data):
    """
    Join the reduced viewing history with Netflix show data based on titles.

    Args:
        reduced_history (np.ndarray): Viewing history data.
        netflix_show_data (np.ndarray): Netflix titles data.

    Returns:
        list: Joined data as a list of dictionaries with keys from both datasets.
    """
    # Reduce the viewing history to relevant titles
    my_titles = set([str(title) for title in reduced_history[:, 0]])

    # Get Netflix show titles
    netflix_titles_dict = {
        str(row[2]): row for row in netflix_show_data
    }  # Use title as key for efficient lookups

    # Perform an inner join
    joined_rows = []

    for row in reduced_history:
        title = str(row[0])
        if title in netflix_titles_dict:
            netflix_row = netflix_titles_dict[title]
            # Combine viewing history row and Netflix data row
            joined_rows.append(np.concatenate((row, netflix_row)))
    
    joined_data = np.array(joined_rows)
    
    try:
        my_found_titles = set([str(title) for title in joined_data[:, 0]])
    except IndexError:  # Handle cases like (0,) or invalid dimensions
        my_found_titles = set()

    print(f"Found Titles: {len(my_found_titles)}")
    print(f"Not Found Titles: {len(my_titles) - len(my_found_titles)}")

    return joined_data


def calculate_show_ratings(viewing_data):
    """
    Calculate ratings for shows based on viewing patterns.

    Args:
        viewing_data (np.ndarray): An array with columns [Show name, Week number, View times].

    Returns:
        dict: A dictionary with show names as keys and ratings as values.

        - If the user has watched >3 episodes of a show in the same week, for multiple weeks, then the show is rated 5 stars.
        - If the user has watched >0 episode of a show in the same week, for three consecutive weeks, then the show is rated 5 stars.
        - If the user has watched >4 episodes of a show in the same week, then the show is rated 4 stars.
        - If the user has watched >4 episodes of a show in different weeks, for multiple weeks, then the show is rated 3 stars.
        - If the user has watched >3 episodes in total, then the show is rated 2 stars.
        Else the show is rated 1 star

    """
    if viewing_data.size == 0:
        return {}

    # Create a new array with the desired data types
    new_viewing_data = np.empty(viewing_data.shape, dtype=object)
    new_viewing_data[:, 0] = viewing_data[:, 0]  # Show name remains as string
    new_viewing_data[:, 1] = viewing_data[:, 1].astype(int)  # Week number as int
    new_viewing_data[:, 2] = viewing_data[:, 2].astype(int)  # View times as int

    # Group data by show name
    show_groups = defaultdict(list)
    for show, week, views in new_viewing_data:
        show_groups[show].append((week, views))

    ratings = {}

    for show, weekly_views in show_groups.items():
        weekly_views = sorted(weekly_views)  # Sort by week number
        total_views = sum(views for _, views in weekly_views)

        high_weeks = sum(1 for _, views in weekly_views if views > 3)
        weeks_with_more_than_one_episode = [week for week, views in weekly_views if views > 1]
        weeks_with_views = [views for _, views in weekly_views if views > 0]

        if high_weeks > 1:
            # Rule 1: If watched >3 episodes in the same week, for multiple weeks -> 5 stars
            ratings[show] = 5
        elif len(weeks_with_views) >= 3 and all(
            weeks_with_more_than_one_episode[i] + 1 == weeks_with_more_than_one_episode[i + 1] == weeks_with_more_than_one_episode[i + 2] - 1
            for i in range(len(weeks_with_more_than_one_episode) - 2)
        ):
            # Rule 2: If watched >1 episode in the same week, for 3 consecutive weeks -> 5 stars
            ratings[show] = 5
        elif any(views > 4 for _, views in weekly_views):
            # Rule 3: If watched >4 episodes in the same week -> 4 stars
            ratings[show] = 4
        elif len(weeks_with_views) > 1 and sum(weeks_with_views) > 4:
            # Rule 4: If watched >4 episodes across different weeks, for multiple weeks -> 3 stars
            ratings[show] = 3
        elif total_views > 3:
            # Rule 5: If watched >3 episodes in total -> 2 stars
            ratings[show] = 2
        else:
            # Rule 6: Else -> 1 star
            ratings[show] = 1

    return ratings

## ==================================================================================================
## Data Processing (3) - Viewing History Aggregation and Storage
## ==================================================================================================

def aggregate_title_week_counts(reduced_data: np.ndarray) -> np.ndarray:
    """
    Aggregate the reduced viewing history by counting occurrences for each title and week.

    Args:
        reduced_data (np.ndarray): A 2D array with titles and weeks.

    Returns:
        np.ndarray: A 2D array with aggregated counts for each title and week combination.
    """
    counts = Counter(map(tuple, reduced_data))
    aggregated_data = np.array([[title, week, str(count)] for (title, week), count in counts.items()])
    return aggregated_data

def save_npy_data(folder, filename, data):
    """
    Save data to the specified path in .npy format.

    Args:
        data: The data to save.
        save_path: Path to save the data.
    """
    np.save(str(folder / filename), data)
    print(f"Data saved to {str(folder / filename)}")
