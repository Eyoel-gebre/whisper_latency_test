import os
import time
import numpy as np
from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
import matplotlib.pyplot as plt

def run_benchmark(llm, num_prompts, num_iterations=3):
    """Run benchmark with given number of prompts and return average latency and throughput."""
    print(f"\nCreating {num_prompts} prompts...")
    try:
        # Create prompts for the batch
        base_prompt = {
            "prompt": "<|startoftranscript|>",
            "multi_modal_data": {
                "audio": AudioAsset("mary_had_lamb").audio_and_sample_rate,
            },
        }
        prompts = [base_prompt] * num_prompts
    except Exception as e:
        print(f"Error creating prompts: {e}")
        raise

    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=200,
    )

    # Run multiple iterations and measure time
    latencies = []
    print(f"Running {num_iterations} iterations...")
    for i in range(num_iterations):
        try:
            print(f"  Iteration {i+1}/{num_iterations}")
            start_time = time.time()
            outputs = llm.generate(prompts, sampling_params)
            end_time = time.time()
            latencies.append(end_time - start_time)
            print(f"    Completed in {latencies[-1]:.2f}s")
        except Exception as e:
            print(f"Error in iteration {i+1}: {e}")
            raise

    avg_latency = np.mean(latencies)
    throughput = num_prompts / avg_latency  # sequences per second

    return avg_latency, throughput

def create_throughput_latency_plot(results):
    """Create and save a plot of throughput vs latency."""
    if not results:
        print("No results to plot!")
        return

    num_prompts, latencies, throughputs = zip(*results)

    # Create figure
    plt.figure(figsize=(10, 6))

    # Line plot connecting the dots
    plt.plot(latencies, throughputs, marker='o', linestyle='-')

    # Label each point with the number of prompts
    for i, num_prompt in enumerate(num_prompts):
        plt.text(latencies[i], throughputs[i], f"num_prompts={num_prompt}", ha='right', va='bottom')

    # Set axis labels and title
    plt.xlabel('Latency (seconds)')
    plt.ylabel('Throughput (sequences/second)')
    plt.title('Throughput vs Latency')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('whisper_benchmark_results.png')
    plt.close()

if __name__ == "__main__":
    # Get model size from user
    model = input('model? (tiny, small, medium, ...) > ')

    # Initialize LLM once
    print("\nInitializing LLM...")
    try:
        llm = LLM(
            model=f"openai/whisper-{model}",
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
        )
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        exit(1)

    # Define numbers of prompts to test (starting small)
    num_prompts_list = [1, 2, 4, 8]  # Start with smaller numbers
    results = []

    print("\nRunning benchmarks...")
    for num_prompts in num_prompts_list:
        print(f"\n{'='*50}")
        print(f"Testing with {num_prompts} prompts")
        print(f"{'='*50}")
        try:
            latency, throughput = run_benchmark(llm, num_prompts)
            results.append((num_prompts, latency, throughput))
            print(f"Success - Latency: {latency:.2f}s, Throughput: {throughput:.2f} sequences/second")
        except Exception as e:
            print(f"\nFailed with {num_prompts} prompts: {str(e)}")
            print("Stopping benchmark run")
            break

    print(f"\nCompleted benchmarks. Total successful runs: {len(results)}")

    if results:
        # Create and save the plot
        create_throughput_latency_plot(results)

        # Print summary table
        print("\nSummary:")
        print("Num Prompts | Latency (s) | Throughput (seq/s)")
        print("-" * 45)
        for num_prompts, latency, throughput in results:
            print(f"{num_prompts:^11d} | {latency:^10.2f} | {throughput:^15.2f}")
    else:
        print("\nNo successful benchmark runs to report!")
        print("Please check the error messages above and verify:")
        print("1. The model name is correct")
        print("2. You have enough GPU memory")
        print("3. The audio file 'mary_had_lamb' is accessible")
        print("4. All required dependencies are installed")
