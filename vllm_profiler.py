# SPDX-License-Identifier: Apache-2.0

import os
import time
from vllm.assets.audio import AudioAsset
from vllm import LLM, SamplingParams

# enable torch profiler, can also be set on cmd line
os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile_2"
num_prompts = 128
# Sample prompts.
base_prompt = {
    "prompt": "<|startoftranscript|>",
    "multi_modal_data": {
    "audio": AudioAsset("mary_had_lamb").audio_and_sample_rate,
    },
}
prompts = [base_prompt] * num_prompts
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

if __name__ == "__main__":

    # Create an LLM.
    llm = LLM(
        model="openai/whisper-large-v3",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )
    llm.start_profile()

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    llm.stop_profile()

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    # Add a buffer to wait for profiler in the background process
    # (in case MP is on) to finish writing profiling output.
    time.sleep(10)
