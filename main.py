import re
import os
import gc
import sys
import json
import time
import torch
import random

from tqdm import tqdm
from datetime import datetime

from engine import GenericAdapter
from task_formatter import TODTaskFormatter
from compute_sims import load_task, find_candidates

gc.enable()
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING']='1'
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'


def parse_waittime(text):
    error_code = int(re.findall(r'Error code: (\d+)', text)[0])
    total_wait = 3
    minutes = seconds = 0
    if error_code == 429:
        time_str = re.findall(r'Please try again in ([0-9ms\.]+)', text)[0]
        try:
            minutes = int(re.findall(r'(\d)m', time_str)[0])
            seconds = int(re.findall(r'(\d+)\.\d*s', time_str)[0])
        except IndexError:
            pass
        total_wait = minutes * 60 + seconds + 1
    return total_wait

persona = 'You are a helpful assistant. Your task is to identify the user\'s intent from a given list of intents and provide the intent name. Provide your answer in a succinct manner.'
persona_prompt = [{'role': 'system', 'content': persona}]

setups = [
    # ('gemma-2-9b-it',
    #  'gemma-2-9b-it',
    #  3),
    # ('gemma-2-9b-it',
    #  'gemma-2-9b-it',
    #  5),
    # ('gemma-2-9b-it',
    #  'gemma-2-9b-it',
    #  10),
    ('mistral-7b-instruct-v0.3',
     'Mistral-7B-Instruct-v0.3',
     3),
    ('mistral-7b-instruct-v0.3',
     'Mistral-7B-Instruct-v0.3',
     5),
    ('mistral-7b-instruct-v0.3',
     'Mistral-7B-Instruct-v0.3',
     10)
]

tasks = ['atis', 'snips', 'clinic150', 'massive']

args = sys.argv
overwrite = '-ow' in args

for (model, model_path, top_k) in setups:
    print(f"Using model {repr(model)} from {repr(model_path)} with k={top_k}")

    adapter = None  # unalloc previous adapter memory
    torch.cuda.empty_cache()

    for task in tasks:
        data, intents, descriptions, labels = load_task(task)
        candidates = find_candidates(task, top_k=top_k, intents=intents)
        output_path = f"{task}-{model}-{top_k}.json"

        output_text = []

        if not overwrite and os.path.exists(output_path):
            output_text = json.load(open(output_path))
            n_existing = len(output_text)
            print(f"Found {n_existing} existing entries")
            data = data[n_existing:]
            labels = labels[n_existing:]
            candidates = candidates[n_existing:]

        if len(data) > 0 and adapter is None:  # Load only if compute necessary
            try:
                adapter = GenericAdapter(model_path=model_path)
            except Exception as e:
                if not isinstance(e, KeyboardInterrupt):
                    print(f"[Out of memory at {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}]")
                    exit(2)

        for i, (entry, candidate_entry) in enumerate(tqdm(list(zip(data, candidates)))):
            utterance = entry['text']
            prompt_text = TODTaskFormatter.format_prompt(candidate_entry, descriptions, utterance)
            prompt = adapter.format_prompt(prompt_text)
            while True:
                try:
                    output = adapter.complete(prompt)
                    out = TODTaskFormatter.format_output(output)
                    break
                except RuntimeError as e:
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
