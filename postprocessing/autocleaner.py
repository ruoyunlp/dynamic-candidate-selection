import os
import re
import sys
import json
import shutil
import itertools

import pandas as pd
import numpy as np

from tqdm import tqdm

from cleaner import clean, score, map_to_intent

root_dir = '.'
results_dir = f"{root_dir}/results"

models = [
    'Meta-Llama-3.1-8B-Instruct',
    'gemma-2-9b-it',
    'Phi-3-medium-4k-instruct',
    'Mistral-7B-Instruct-v0.3',
]

model_to_name = {
    'llama-3.1-8b-instruct': 'Meta-Llama-3.1-8B-Instruct',
    'gemma-2-9b-it': 'gemma-2-9b-it',
    'phi-3-medium-4k-instruct': 'Phi-3-medium-4k-instruct',
    'mistral-7b-instruct-v0.3': 'Mistral-7B-Instruct-v0.3'
}

abbreviations = {
    'mistral-7b-instruct-v0.3': 'mistral-7b-instruct',
    '3': '',
}

def abbrvname(name):
    if name in abbreviations: return abbreviations[name]
    elif re.match(r'\d+', name): return f"_k{name}"
    return name

def get_k_from_quant(quant):
    matches = re.findall(r'_k(\d+)_?', quant)
    if len(matches) == 0: return 3
    return int(matches[0])

def recursive_clean(configs=None, do_clean=True, do_eval=False, do_purge=False):
    # set args to use later
    args = []
    if do_clean: args.append('clean')
    if do_eval: args.append('eval')

    # save eval into csv or xlsx file for easy trasnfer of bulk results
    if do_eval: results = {}

    # Create configs and clean all at the same time
    if configs is None:
        configs = []
        for model in models:
            model_dir = os.path.join(results_dir, model)
            quantizations = os.listdir(model_dir)
            for quant in quantizations:
                results_dir_raw = os.path.join(model_dir, quant, 'raw')
                results_dir_clean = os.path.join(model_dir, quant, 'clean')

                # Purge existing clean directory
                if do_purge: shutil.rmtree(results_dir_clean, ignore_errors=True)

                task_files = os.listdir(results_dir_raw)
                for f_task in task_files:
                    (task, model_name) = re.findall(r'([^-]+)-([\S]+).json', f_task)[0]

                    configs.append((task, quant, model_name))

    # clean and/or evaluate only the specified combinations
    if do_clean or do_eval:
        for (task, quant, model_name) in tqdm(configs):
            top_k = get_k_from_quant(quant)
            candidates = json.load(open(f"{root_dir}/candidates/{task}-bge-large-en-v1.5-e+p+om-{top_k}-cands.jsonl"))
            out_clean = clean(args + [task], quant, model_name, do_print=False, candidates=candidates)
            print(f"Cleaned: {(task, quant, model_name)}")
            if do_eval: results[(model_name, quant, task)] = out_clean

    if do_eval:
        with open(f"{results_dir}/auto-clean-out.jsonl", 'w') as output_file:
            output_file.write(json.dumps(results))

def iterative_eval(setups=None, models=None, quantizations=None, top_ks=None):
    def parse(pattern, text):
        return len(re.findall(pattern, text)) > 0


    if setups is None:
        if top_ks is not None:
            quantizations = list(itertools.product(quantizations, top_ks))
            quantizations = [q + abbrvname(k) for (q, k) in quantizations]
        setup_combos = [models, quantizations]
        setups = list(itertools.product(*setup_combos))

    results = []
    tasks = ['atis', 'snips', 'clinic150', 'massive']
    for (model, quant) in tqdm(setups):

        top_k = get_k_from_quant(quant)
        encoder = 'bge-large-en-v1.5'
        cs_setup = 'e+p+om'

        if parse(r'_gte_', quant): encoder = 'gte-large'
        if parse(r'_ecs', quant): cs_setup = 'e'

        setup_out = {
            "model": model,
            "quantization": quant,
            "top-k": top_k
        }

        for task in tasks:
            task_acc = f"{task}-acc"
            task_f1 = f"{task}-f1"
            task_mean = f"{task}-mean"

            model_output_clean = f"{results_dir}/{model_to_name[model]}/{quant}/clean/{task}-{abbrvname(model)}.json"
            if not os.path.exists(model_output_clean):
                setup_out[task_acc] = None
                setup_out[task_f1] = None
            else:
                model_output_clean = json.load(open(model_output_clean))
                all_candidates = json.load(open(f"{root_dir}/candidates/{task}-{encoder}-{cs_setup}-{top_k}-cands.jsonl"))
                intents = open(f"{root_dir}/data/{task}/intents.txt").read().splitlines()
                labels = []
                preds = []
                for (entry, candidates) in zip(model_output_clean, all_candidates):
                    labels.append(intents.index(entry['label']))
                    preds.append(intents.index(map_to_intent(entry['model_out'], candidates)))
                labels = np.array(labels)
                preds = np.array(preds)
                setup_out[task_acc], setup_out[task_f1] = score(labels, preds)

            setup_out[task_mean] = None

        results.append(setup_out)

    results = pd.DataFrame(results)
    results.to_csv('results_full.csv')


if __name__ == "__main__":
    args = sys.argv[1:]
    do_clean = '-clean' in args
    do_purge = '-purge' in args
    do_eval = '-eval' in args

    recursive_clean(do_clean=do_clean, do_eval=do_eval, do_purge=do_purge)
