import os
from datetime import datetime, date

API_NAME = os.getenv("API_NAME")

def is_file_modified_today(file_path):
    """
    Checks if a file was created or modified today.

    :param file_path: Path to the file
    :return: True if the file was created or modified today, False otherwise
    """
    if not os.path.exists(file_path):
        print(f"The file '{file_path}' does not exist.")
        return False

    # Get the file's last modification time
    file_mod_time = os.path.getmtime(file_path)
    
    # Convert the timestamp to a date
    file_mod_date = datetime.fromtimestamp(file_mod_time).date()

    # Compare with today's date
    return file_mod_date == date.today()


def should_run(interval=1) -> bool:
    INTERVAL = interval
    timestamp_file = f"./script_timestamps/{API_NAME}_last_run"
    os.makedirs(os.path.dirname(timestamp_file), exist_ok=True)
    now = datetime.now().timestamp()
    time_diff = INTERVAL  # default to running if no file exists
    if os.path.exists(timestamp_file):
        try:
            with open(timestamp_file, "r") as f:
                last_run = int(f.read().strip())
                time_diff = now - last_run
        except (FileNotFoundError, ValueError):
            print(f"Unable to read timestamp file: {timestamp_file}")
    if time_diff >= INTERVAL:
        with open(timestamp_file, "w") as f:
            f.write(f"{int(now)}")
        return True
    return False