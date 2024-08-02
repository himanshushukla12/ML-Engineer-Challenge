import os
import requests
import zipfile

def download_tiny_imagenet(url, dataset_path='tiny-imagenet-200'):
    # Define the download path
    download_path = f'{dataset_path}.zip'

    # Download the dataset
    print(f'Downloading {url}...')
    response = requests.get(url, stream=True)
    with open(download_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=128):
            file.write(chunk)
    
    # Extract the dataset
    print(f'Extracting {download_path}...')
    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_path)
    
    # Clean up the zip file
    os.remove(download_path)
    print(f'Dataset downloaded and extracted to {dataset_path}')

# Example usage
if __name__ == "__main__":
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    download_tiny_imagenet(url)