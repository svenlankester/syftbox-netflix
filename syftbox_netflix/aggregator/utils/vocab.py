import json
import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from syftbox_netflix.aggregator.utils.logging_setup import logger

print(logger)

def create_tvseries_vocab(shared_folder: Path, zip_file_path = None):
    # TODO: retrieve most up-to-date file, currently is loading a static file with Netflix series

    if not zip_file_path:
        zip_file = os.path.join(os.getcwd(), "syftbox_netflix", "aggregator", "data", "netflix_series_2024-12.csv.zip")
    else:
        zip_file = zip_file_path

    logging.debug(f"[vocab.py] Loading {zip_file}...")
    
    df = pd.read_csv(zip_file)

    label_encoder = LabelEncoder()
    label_encoder.fit(df['Title'])

    vocab_mapping = {title: idx for idx, title in enumerate(label_encoder.classes_)}
    
    output_path = Path(shared_folder) / "tv-series_vocabulary.json"
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(vocab_mapping, f, ensure_ascii=False, indent=4)
        
    return vocab_mapping