import torch
import time
import argparse
from math import ceil
from vllm import LLM, SamplingParams
from datasets import load_dataset
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
cur_time = int(time.time())
vllm_profile_loc = f"vllm_profile_{cur_time}"
os.environ["VLLM_TORCH_PROFILER_DIR"] = vllm_profile_loc

def run_profiling(batch_size, num_samples, dataset):
    llm = LLM(
        model="openai/whisper-large-v3",
        max_model_len=448,
        max_num_seqs=batch_size,
        limit_mm_per_prompt={"audio": 1},
        kv_cache_dtype="fp8", 
    )

    dataset = load_dataset('esb/diagnostic-dataset', dataset)
    SAMPLING_RATE = 16000

    # Collect samples with their transcription lengths
    samples_with_lengths = []
    count = 0
    for audio_sample in dataset["clean"]:
        audio_array = audio_sample["audio"]['array']
        transcription = audio_sample["norm_transcript"]
        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "audio": (audio_array, SAMPLING_RATE),
                },
            },
            "decoder_prompt": "<|startoftranscript|>",
        }
        samples_with_lengths.append((len(transcription), prompt))
        count += 1
        if count >= num_samples:
            break
    
    # Sort by transcription length and extract just the prompts
    # samples_with_lengths.sort(key=lambda x: x[0])
    prompts = [sample[1] for sample in samples_with_lengths]

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=200,
    )

    times = []
    throughputs = []
    for i in range(20):
        # llm.start_profile()
        start = time.perf_counter()
        llm.generate(prompts, sampling_params)
        end = time.perf_counter()
        times.append(end - start)
        throughputs.append(len(prompts) / (end - start))
        print(f"Time taken: {end - start} seconds for {i+1}th iteration")
        print(f"Throughput: {throughputs[-1]} samples/second")
        # llm.stop_profile()

    # print(f"Profiling data saved to: {vllm_profile_loc}")
    print(f"Average time taken: {sum(times) / len(times)} seconds")
    print(f"Average throughput: {sum(throughputs) / len(throughputs)} requests/second")
    print('times: ', times)
    print('throughputs: ', throughputs)
    time.sleep(5)
    print("Done!")

if __name__ == "__main__":
    dataset = input("Enter the dataset name: ")
    batch_size = int(input("Enter the batch size: "))
    num_samples = int(input("Enter the number of samples: "))
    run_profiling(batch_size, num_samples, dataset)
