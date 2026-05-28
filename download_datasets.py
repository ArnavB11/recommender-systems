import os
import urllib.request
import zipfile
import tarfile

def download_and_extract(url, extract_to):
    print(f"Downloading {url}...")
    filename = url.split('/')[-1]
    filepath = os.path.join(extract_to, filename)
    
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
        
    if not os.path.exists(filepath):
        urllib.request.urlretrieve(url, filepath)
        print(f"Downloaded to {filepath}")
    else:
        print(f"File {filepath} already exists, skipping download.")
    
    print(f"Extracting {filepath}...")
    if filepath.endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif filepath.endswith('.tar.gz'):
        with tarfile.open(filepath, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    
    print("Done extracting.")

if __name__ == "__main__":
    data_dir = "data"
    
    # MovieLens 100k
    ml_url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    download_and_extract(ml_url, data_dir)
    
    # MovieLens Latest Small (needed for IMDb ID linking)
    ml_small_url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    download_and_extract(ml_small_url, data_dir)
    
    # IMDB dataset
    imdb_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    download_and_extract(imdb_url, data_dir)
    
    print("All datasets downloaded and extracted successfully.")
