import os
import sys
import subprocess
from syftbox.lib import Client
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

API_NAME = os.getenv("API_NAME")
AGGREGATOR_DATASITE = os.getenv("AGGREGATOR_DATASITE")
CSV_NAME = os.getenv("NETFLIX_CSV")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

## ==================================================================================================
## Application functions
## ==================================================================================================
def run_execution_context(client):
    """
    Determine and handle execution context (aggregator vs. participant).
    """
    if client.email == AGGREGATOR_DATASITE:
        print(f">> {API_NAME} | Running as aggregator.")
        subprocess.run([sys.executable, "aggregator/main.py"])
        sys.exit(0)
    else:
        print(f">> {API_NAME} | Running as participant.")
        subprocess.run([sys.executable, "participant/main.py"])
        sys.exit(0)

## ==================================================================================================
## Orchestrator
## ==================================================================================================
def main():
    # Load client and run execution context
    client = Client.load()
    run_execution_context(client)

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
