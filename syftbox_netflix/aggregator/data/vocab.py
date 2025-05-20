import json
from pathlib import Path

aggregator_data = Path(__file__).resolve().parent


def get_local_vocabulary_path():
    local_path = aggregator_data / "tv-series_vocabulary.json"
    return local_path


def get_local_vocabulary_json():
    with open(
        get_local_vocabulary_path(),
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)
