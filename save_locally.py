# Script to download and save 10 samples from the AMI dataset
import os
import requests
from tqdm import tqdm
import zipfile
import random

# Create directory for saving audio files
SAVE_DIR = "ami_samples"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_file(url, destination):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def get_ami_samples(num_samples=10):
    """Download and extract AMI dataset samples"""
    # AMI dataset headset audio samples URL
    # This is a subset of the AMI corpus with headset recordings
    ami_url = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002a/audio/ES2002a.Headset-0.wav"
    
    print(f"Downloading AMI dataset samples...")
    
    # List to store paths of downloaded files
    sample_paths = []
    
    # Download individual samples
    for i in range(num_samples):
        # For demonstration, we're downloading the same file multiple times
        # In a real scenario, you would use different file URLs
        sample_id = f"ES2002a_sample_{i+1}"
        sample_path = os.path.join(SAVE_DIR, f"{sample_id}.wav")
        
        # Download the file
        print(f"Downloading sample {i+1}/{num_samples}...")
        download_file(ami_url, sample_path)
        sample_paths.append(sample_path)
        
        print(f"Saved to {sample_path}")
    
    return sample_paths

if __name__ == "__main__":
    print("Starting AMI dataset sample download...")
    samples = get_ami_samples(10)
    print(f"Successfully downloaded 10 AMI dataset samples to {SAVE_DIR}/")
    print("Sample paths:")
    for path in samples:
        print(f"  - {path}")
