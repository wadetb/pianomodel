import os
import requests

MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
DATA_DIR = "."
ARCHIVE_NAME = "maestro-v3.0.0.zip"

def download_maestro_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    archive_path = os.path.join(DATA_DIR, ARCHIVE_NAME)
    if not os.path.exists(archive_path):
        print(f"Downloading MAESTRO 3.0.0 piano dataset from {MAESTRO_URL}...")
        with requests.get(MAESTRO_URL, stream=True) as r:
            r.raise_for_status()
            with open(archive_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")
    else:
        print("Archive already exists.")
    print("Extracting dataset...")
    os.system(f"unzip {archive_path} -d {DATA_DIR}")
    print("Extraction complete.")

if __name__ == "__main__":
    download_maestro_dataset()
