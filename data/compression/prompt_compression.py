import re
import os
import gc
import sys
import json
import time
import torch
import random

import numpy as np

from tqdm import tqdm
from datetime import datetime
from llmlingua import PromptCompressor

from engine import GenericAdapter
from task_formatter import TODTaskFormatter, TODCoTTaskFormatter
from compute_sims import load_task, find_candidates
from compression.utils import build_compressed_prompt

gc.enable()
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING']='1'
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'


def preproc_cot(top_k=151):
    ROOT_DIR = '.'
    tasks = ['atis', 'snips', 'clinic150', 'massive']
    for task in tasks:
        compressed_prompts = []
        data, intents, descriptions, labels = load_task(task)
        all_candidates = [intents for _ in range(len(data))]
        output_path = f"{ROOT_DIR}/compression/{task}-cot.json"

        if top_k != 151:
            all_candidates = find_candidates(task, top_k=top_k, intents=intents)
            output_path = f"{ROOT_DIR}/compression/{task}-cot-{top_k}.json"

        if os.path.exists(output_path):
            compressed_prompts = json.load(open(output_path))
            n_existing = len(compressed_prompts)
            data = data[n_existing:]
            all_candidates = all_candidates[n_existing:]

        for i, (entry, candidates) in enumerate(tqdm(list(zip(data, all_candidates)))):
            utterance = entry['text']
            prompt_text = TODCoTTaskFormatter.format_prompt(candidates, descriptions, utterance)
            compressed_prompts.append({
                "index": entry['index'],
                "text": entry['text'],
                "intent": entry['intent'],
                "prompt": prompt_text
            })

        with open(output_path, 'w') as output_file:
            output_file.write(json.dumps(compressed_prompts, indent=4))

def run_compression(overwrite=False):
    model_type = 'microsoft/llmlingua-2-xlm-roberta-large-meetingbank'
    ROOT_DIR = ''
    tasks = ['clinic150', 'massive']
    # MODEL_DIR = f"{ROOT_DIR}/{model_type}"
    compressor = PromptCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        use_llmlingua2=True
    )
    for task in tasks:
        compressed_prompts = []
        compression_stats = []
        data, intents, descriptions, labels = load_task(task)
        output_path = f"{ROOT_DIR}/compression/{task}-compressed.json"
        if os.path.exists(output_path):
            compressed_prompts = json.load(open(output_path))
            n_existing = len(compressed_prompts)
            data = data[n_existing:]
            labels = data[n_existing:]

        for i, entry in enumerate(tqdm(data)):
            utterance = entry['text']
            prompt_text = TODTaskFormatter.format_prompt(intents, descriptions, utterance)
            prompt_compressed = compressor.compress_prompt_llmlingua2(
                prompt_text,
                rate=0.6,
                force_tokens=['\n', '.', '!', '?', ','],
                chunk_end_tokens=['.', '\n'],
                return_word_label=True,
                drop_consecutive=True
            )
            compressed_prompts.append({
                "index": entry['index'],
                "text": entry['text'],
                "intent": entry['intent'],
                "prompt": prompt_compressed['compressed_prompt']
            })
            with open(output_path, 'w') as output_file:
                output_file.write(json.dumps(compressed_prompts, indent=4))


def run_classification():
    persona = 'You are a helpful assistant. Your task is to identify the user\'s intent from a given list of intents and provide the intent name. Provide your answer in a succinct manner.'
    persona_prompt = [{'role': 'system', 'content': persona}]

    setups = [
        ('llama-3.1-8b-instruct',
        'Meta-Llama-3.1-8B-Instruct',
        5,
        '4bit'),
        ('gemma-2-9b-it',
        'gemma-2-9b-it',
        5,
        '8bit'),
        ('phi-3-medium-4k-instruct',
        'Phi-3-medium-4k-instruct',
        5,
        '8bit'),
        ('mistral-7b-instruct-v0.3',
        'Mistral-7B-Instruct-v0.3',
        5,
        '8bit'),
        # ('mistral-7b-instruct-v0.3',
        #  'Mistral-7B-Instruct-v0.3',
        #  5),
        # ('mistral-7b-instruct-v0.3',
        #  'Mistral-7B-Instruct-v0.3',
        #  10),
        # ('phi-3-medium-4k-instruct',
        #  'Phi-3-medium-4k-instruct',
        #  5),
        # ('phi-3-medium-4k-instruct',
        #  'Phi-3-medium-4k-instruct',
        #  10)
    ]

    tasks = ['atis', 'snips']
    # tasks = ['atis', 'snips', 'clinic150', 'massive']

    args = sys.argv
    overwrite = '-ow' in args

    with torch.no_grad():
        for (model, model_path, top_k, quantization) in setups:
            print(f"Using model {repr(model)} from {repr(model_path)} with k={top_k}")

            adapter = None  # unalloc previous adapter memory
            torch.cuda.empty_cache()

            for task in tasks:
                data, intents, descriptions, labels = load_task(task)
                candidates = find_candidates(task, top_k=top_k, intents=intents)
                compressed_prompts = build_compressed_prompt(task, candidates, '.')
                output_path = f"{task}-{model}-{top_k}c.json"

                output_text = []

                if not overwrite and os.path.exists(output_path):
                    output_text = json.load(open(output_path))
                    n_existing = len(output_text)
                    print(f"Found {n_existing} existing entries")
                    data = data[n_existing:]
                    compressed_prompts = compressed_prompts[n_existing:]
                    labels = labels[n_existing:]

                if len(data) > 0 and adapter is None:  # Load only if compute necessary
                    try:
                        adapter = GenericAdapter(model_path=model_path, quantization=quantization)
                    except Exception as e:
                        if not isinstance(e, KeyboardInterrupt):
                            print(e)
                            print(f"[Out of memory at {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}]")
                            exit(2)

                for i, (entry, compressed_prompt) in enumerate(tqdm(list(zip(data, compressed_prompts)))):
                    utterance = entry['text']
                    prompt_text = compressed_prompt
                    if isinstance(compressed_prompt, dict) and 'prompt' in prompt_text:
                        prompt_text = prompt_text['prompt']
                    prompt = adapter.format_prompt(prompt_text)
                    while True:
                        try:
                            output = adapter.complete(prompt)
                            out = TODTaskFormatter.format_output(output)
                            break
                        except RuntimeError as e:
                            print(e)
                            print(f"[Out of memory at {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}]")
                            exit(2)

                    output_text.append({
                        "index": entry['index'],
                        "text": utterance,
                        "prompt_text": prompt_text,
                        "model_out": out,
                        'label': entry['intent']
                    })
                    with open(output_path, 'w') as output_file:
                        output_file.write(json.dumps(output_text))

if __name__ == "__main__":
    run_classification()