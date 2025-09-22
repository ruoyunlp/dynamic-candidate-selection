# ==============================================================================
# NOTE: This file requires the python3.10 environment or later as gemma-3-1b-it
# requires a newer version of the transformers library than is feasible for
# python3.8
# ==============================================================================
import json
import os
import pandas as pd
import re
import sys
import torch

from argparse import ArgumentParser
from Levenshtein import distance
from sklearn.metrics import accuracy_score, f1_score
from string import Template
from tqdm import tqdm

from engine import GenericAdapter

ROOT_DIR = '.'
PROMPT = Template("""
You are a part of a task-oriented dialogue model.
Your task is, given a model output (sentence) that is trying to identify an intent, give the intent name that was identified.
The sentence is '$sentence'
Please give the intent name only. Do not give any other information or reasoning.
The intent is
""")


setups = {
    'llama-3.1-8b-instruct': [
        '4_bit_quant',
        '4_bit_quant_cot',
        '4_bit_quant_llmlingua2',
        '4_bit_quant_k5',
        '4_bit_quant_k5_llmlingua2',
        '4_bit_quant_k5_llmlingua2_recon',
        '4_bit_quant_dynamick_matmul_plus4',
        '4_bit_quant_k5_rerun1',
        '4_bit_quant_k5_rerun2',
        '4_bit_quant_k5_rerun3',
        '4_bit_quant_k10',
        '8_bit_quant',
        '8_bit_quant_k5',
        '8_bit_quant_k10',
        '8_bit_quant_k16',
        '8_bit_quant_k32',
        '8_bit_quant_k64',
        '8_bit_quant_k75',
        '8_bit_quant_k128',
        '8_bit_quant_k151',
        'full',
    ],
    'gemma-2-9b-it': [
        '4_bit_quant',
        '4_bit_quant_k5',
        '4_bit_quant_k10',
        '8_bit_quant',
        '8_bit_quant_cot',
        '8_bit_quant_k5',
        '8_bit_quant_k10',
        '8_bit_quant_k5_cot',
        '8_bit_quant_llmlingua2',
        '8_bit_quant_k5_llmlingua2',
        '8_bit_quant_k5_llmlingua2_recon',
        '8_bit_quant_dynamick_matmul_plus4',
        '8_bit_quant_k5_rerun1',
        '8_bit_quant_k5_rerun2',
        '8_bit_quant_k5_rerun3',
        '8_bit_quant_k16',
        '8_bit_quant_k32',
        '8_bit_quant_k64',
        '8_bit_quant_k75',
        '8_bit_quant_k128',
        '8_bit_quant_k151',
        'full',
    ],
    'phi-3-medium-4k-instruct': [
        '4_bit_quant',
        '4_bit_quant_k5',
        '4_bit_quant_k10',
        '8_bit_quant',
        '8_bit_quant_k5',
        '8_bit_quant_k10',
        '8_bit_quant_k151',
        '16_bit_quant',
    ],
    'mistral-7b-instruct': [
        '4_bit_quant',
        '4_bit_quant_k5',
        '4_bit_quant_k10',
        '8_bit_quant',
        '8_bit_quant_k5',
        '8_bit_quant_k10',
        '8_bit_quant_k151',
        '8_bit_quant_dynamick_matmul_plus4',
        'full',
    ],
}

model_to_name = {
    'llama-3.1-8b-instruct': 'Meta-Llama-3.1-8B-Instruct',
    'gemma-2-9b-it': 'gemma-2-9b-it',
    'phi-3-medium-4k-instruct': 'Phi-3-medium-4k-instruct',
    'mistral-7b-instruct': 'Mistral-7B-Instruct-v0.3'
}

all_tasks = ['atis', 'snips', 'clinic150', 'massive']

compressor_configs = [
    ('gemma-2-9b-it',
     'gemma-2-9b-it',
     '4bit',
     'lmclean2'),
    # ('gemma-2-2b-it',
    #  'gemma-2-9b-it',
    #  '4bit',
    #  'lmclean3'),
    # ('gemma-3-1b-it',
    #  'gemma-3-1b-it',
    #  '16bit',
    #  'lmclean')
]

def score(labels, preds, do_print=True, tag=None):
    acc = accuracy_score(labels, preds) * 100
    f1 = f1_score(labels, preds, average='macro') * 100
    if do_print:
        tag = f"[{tag}]" if tag is not None else ''
        print(f"{tag} acc: {(acc):.4f} macro-f1: {(f1):.4f}")
    return acc, f1

def map_to_intent(text, intents):
    distances = torch.tensor([distance(text, intent) for intent in intents])
    closest_match = intents[distances.argmin(dim=0)]
    return closest_match

def do_clean(args):
    for (model, model_path, model_quant, lmclean) in compressor_configs:
        adapter = GenericAdapter(model_path, model_quant)

        for model_type in setups:
            for quantization in setups[model_type]:
                results_dir = f"{ROOT_DIR}/results/{model_to_name[model_type]}/{quantization}"
                print(f"[Cleaning '{model_type}' '{quantization}']")
                for task in tasks:
                    intents = open(f"{ROOT_DIR}/data/{task}/intents.txt").read().splitlines()

                    input_path = f"{results_dir}/raw/{task}-{model_type}.json"
                    output_dir = f"{results_dir}/{lmclean}"
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    output_path = f"{output_dir}/{task}-{model_type}.jsonl"

                    if os.path.exists(input_path):
                        results = json.load(open(input_path))
                        model_output_cleaned = []
                        write_mode = 'w'
                        if os.path.exists(output_path):
                            if args.overwrite:
                                print(f"Overwriting {output_path}")
                                os.remove(output_path)
                            else:
                                model_output_cleaned = open(output_path).read().splitlines()
                                model_output_cleaned = [json.loads(entry) for entry in model_output_cleaned]
                                n_existing = len(model_output_cleaned)
                                print(f"Found {n_existing} existing entries in {output_path}")
                                results = results[n_existing:]
                                write_mode = 'a'

                        if len(results) > 0:
                            with open(output_path, write_mode) as output_file:
                                results_iterator = tqdm(results)
                                results_iterator.set_description(task.upper())
                                for entry in results_iterator:
                                    torch.cuda.empty_cache()  # clear cache
                                    model_out = entry['model_out'].strip()
                                    out = model_out
                                    if out not in intents:  # try selecting last word first
                                        out = re.findall(r'([^\s\.\"\'\|]+)[\s\.\"\'\|]*$', model_out)[-1]
                                        # out = re.findall('\S+', model_out)[-1]
                                    if out not in intents:  # use lm to extract prediction
                                        model_out = re.sub('\s+', ' ', model_out)
                                        prompt = PROMPT.substitute({'sentence': model_out}).strip()
                                        prompt = adapter.format_prompt(prompt)
                                        out, _ = adapter.complete(prompt)

                                    model_output_cleaned.append({
                                        'index': entry['index'],
                                        'text': entry['text'],
                                        'prompt_text': entry['prompt_text'],
                                        'model_out': entry['model_out'],
                                        'cleaned': out.strip(),
                                        'label': entry['label']
                                    })
                                    output_file.write(json.dumps(model_output_cleaned[-1]) + '\n')
                    else:
                        print(f"Unable to find file at '{input_path}'")

def do_mapping(args):
    output_df = []
    for (extractor, _, _, lmclean) in compressor_configs:
        for model_type in setups:
            for quantization in setups[model_type]:
                output_entry = {
                    'model_type': model_type,
                    'quantization': quantization,
                    'extractor': extractor
                }

                print(f"[Checking {repr(model_type)} {repr(quantization)}]")
                for task in tasks:
                    intents = open(f"{ROOT_DIR}/data/{task}/intents.txt").read().splitlines()
                    results_dir = f"{ROOT_DIR}/results/{model_to_name[model_type]}/{quantization}"
                    cleaned_path = f"{results_dir}/{lmclean}/{task}-{model_type}.jsonl"
                    output_path = f"{results_dir}/clean/{task}-{model_type}.json"

                    if os.path.exists(cleaned_path):
                        model_output_cleaned = open(cleaned_path).read().splitlines()
                        model_output_cleaned = [json.loads(entry) for entry in model_output_cleaned]
                        model_output_mapped = []
                        labels = []
                        preds = []

                        failed_maps = 0
                        for entry in model_output_cleaned:
                            out = entry['cleaned'].strip()
                            if out not in intents:
                                failed_maps += 1
                                out = map_to_intent(out, intents)

                            labels.append(intents.index(entry['label']))
                            preds.append(intents.index(out.strip()))

                            model_output_mapped.append({
                                'index': entry['index'],
                                'text': entry['text'],
                                'prompt_text': entry['prompt_text'],
                                'model_out': out,
                                'label': entry['label']
                            })
                        print(f"{task.upper()} Failed to clean {failed_maps}/{len(model_output_cleaned)} examples")

                        labels = torch.tensor(labels)
                        preds = torch.tensor(preds)
                        acc, f1 = score(labels, preds)

                        # log cleaned results
                        output_entry[task.upper() + '-acc'] = acc
                        output_entry[task.upper() + '-f1'] = f1
                        output_entry[task.upper() + '-ovr'] = (acc + f1) / 2

                        with open(output_path, 'w') as output_file:
                            output_file.write(json.dumps(model_output_mapped, indent=5))

                    else:
                        print(f"{task.upper()} Unable to find lmcleaned file at {repr(cleaned_path)}")

                output_df.append(output_entry)

    output_df = pd.DataFrame(output_df)
    output_df.to_excel("results-cleaned-overall.xlsx", index=False)

args = sys.argv[1:]
tasks = all_tasks

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-s', '--stages', nargs='+', required=True)
    parser.add_argument('-ow', '--overwrite', action='store_true')

    args = parser.parse_args()
    print(args.__dict__)
    if 'lmclean' in args.stages:
        do_clean(args)
    if 'mapping' in args.stages:
        do_mapping(args)

