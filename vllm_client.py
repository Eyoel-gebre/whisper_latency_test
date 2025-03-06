# SPDX-License-Identifier: Apache-2.0
from openai import OpenAI
import os
import glob

# Path to the directory containing AMI samples
AMI_SAMPLES_DIR = "ami_samples"

# Check if samples exist
sample_files = glob.glob(os.path.join(AMI_SAMPLES_DIR, "*.wav"))
if not sample_files:
    raise FileNotFoundError(
        f"No audio samples found in {AMI_SAMPLES_DIR}. "
        "Please run save_locally.py first to download the samples."
    )

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# Process each sample
for i, sample_path in enumerate(sample_files[:10], 1):
    print(f"Processing sample {i}/{len(sample_files[:10])}: {os.path.basename(sample_path)}")
    
    with open(sample_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model="openai/whisper-large-v3",
            language="en",
            response_format="text",
            temperature=0.0)
        
        print(f"Transcription result for {os.path.basename(sample_path)}:")
        print(transcription)
        print("-" * 80)