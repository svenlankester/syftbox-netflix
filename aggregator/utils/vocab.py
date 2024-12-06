import json
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

def create_tvseries_vocab(shared_folder: Path):
    zip_file = os.path.join(os.getcwd(), "aggregator", "data", "netflix_series_2024-12.csv.zip")  # TODO: retrieve most up-to-date file
    df = pd.read_csv(zip_file)

    label_encoder = LabelEncoder()
    label_encoder.fit(df['Title'])

    vocab_mapping = {title: idx for idx, title in enumerate(label_encoder.classes_)}
    
    output_path = os.path.join(str(shared_folder), "tv-series_vocabulary.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_mapping, f, ensure_ascii=False, indent=4)