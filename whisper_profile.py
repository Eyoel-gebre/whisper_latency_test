import torch
import time
import argparse
from math import ceil
from vllm import LLM, SamplingParams
from datasets import load_dataset
import os
    
# Set up profiling directory
cur_time = int(time.time())
vllm_profile_loc = f"vllm_profile_{cur_time}"
os.environ["VLLM_TORCH_PROFILER_DIR"] = vllm_profile_loc
    

def run_profiling(batch_size, num_samples, dataset):
    # Initialize the Whisper model via vLLM
    print(f"Initializing Whisper model...")
    llm = LLM(
        model="openai/whisper-large-v3",
        max_model_len=448,
        max_num_seqs=batch_size,
        limit_mm_per_prompt={"audio": 1},
        kv_cache_dtype="fp8", 
    )

    # Load dataset
    print(f"Loading dataset {dataset}...")
    dataset = load_dataset('esb/diagnostic-dataset', dataset)
    SAMPLING_RATE = 16000  # Same as in whisper_evaluation_server.py
    
    # Prepare prompts
    print(f"Preparing {num_samples} prompts...")
    prompts = []
    count = 0
    for audio_sample in dataset["clean"]:
        audio_array = audio_sample["audio"]['array']
        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "audio": (audio_array, SAMPLING_RATE),
                },
            },
            "decoder_prompt": "<|startoftranscript|>",
        }
        prompts.append(prompt)
        count += 1
        if count >= num_samples:
            break
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,  # Fixed value instead of args.temperature
        top_p=1.0,        # Fixed value instead of args.top_p
        max_tokens=200,   # Fixed value instead of args.max_tokens
    )
    
    # Calculate batches
    total_batches = ceil(len(prompts) / batch_size)
    
    # Run profiling
    print(f"Starting profiling with {total_batches} batches (batch size: {batch_size})...")
    
    # Start profiling
    llm.start_profile()
    
    # Process batches
    #for batch_idx in range(total_batches):
        #batch_prompts = prompts[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        #
        #print(f"Processing batch {batch_idx+1}/{total_batches} with {len(batch_prompts)} samples...")
        #batch_start = time.time()
        #outputs = llm.generate(batch_prompts, sampling_params)
        #batch_duration = time.time() - batch_start
        #
        #print(f"Batch {batch_idx+1} completed in {batch_duration:.2f} seconds")
    llm.generate(prompts, sampling_params)
    
    # Stop profiling
    llm.stop_profile()
    print(f"Profiling data saved to: {vllm_profile_loc}")
    
    # Wait for profiler to write data
    print("Waiting for profiler to write trace data...")
    time.sleep(5)
    print("Done!")

if __name__ == "__main__":
    dataset = input("Enter the dataset name: ")
    batch_size = int(input("Enter the batch size: "))
    num_samples = int(input("Enter the number of samples: "))
    run_profiling(batch_size, num_samples, dataset)
