import os
import subprocess
import sys
from datetime import datetime

from dotenv import load_dotenv
from syft_core import Client as SyftboxClient
from syft_core import SyftClientConfig

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

API_NAME = os.getenv("API_NAME", "syftbox-netflix")
AGGREGATOR_DATASITE = os.getenv("AGGREGATOR_DATASITE")
CSV_NAME = os.getenv("NETFLIX_CSV")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
NETFLIX_PROFILE = os.getenv("NETFLIX_PROFILE", "PLACEHOLDER_PROFILE")
NETFLIX_PROFILES = os.getenv("NETFLIX_PROFILES", NETFLIX_PROFILE)


def should_run(interval=20) -> bool:
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


## ==================================================================================================
## Application functions
## ==================================================================================================
def run_execution_context(client):
    """
    Determine and handle execution context (aggregator vs. participant).
    """
    if client.email == AGGREGATOR_DATASITE:
        # Skip execution if conditions are not met
        if not should_run(60):
            print(f"Skipping {API_NAME} as Aggregator, not enough time has passed.")
            exit(0)

        print(f">> {API_NAME} | Running as aggregator.")
        subprocess.run(
            [sys.executable, "aggregator/main.py"],
            env=os.environ.copy(),
        )
        print(f">> {API_NAME} | Aggregator execution complete.")

    else:
        # Run participant (aggregator is a participant too)
        # Skip execution if conditions are not met
        # if not should_run(interval=1):
        #     print(f"Skipping {API_NAME} as Participant, not enough time has passed.")
        #     sys.exit(0)
        print(f">> {NETFLIX_PROFILES} ")
        for profile_id, profile in enumerate(NETFLIX_PROFILES.split(",")):
            print(
                f">> {API_NAME} | Running as participant with profile_id: {profile_id}."
            )
            from syftbox_netflix.main import main as participant_main

            participant_main(profile, profile_id)
        sys.exit(0)


## ==================================================================================================
## Orchestrator
## ==================================================================================================
def main():
    # Load client and run execution context
    config = SyftClientConfig.load()
    client = SyftboxClient(config)
    run_execution_context(client)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
