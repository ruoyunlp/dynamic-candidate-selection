import json
import torch
import random

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from Levenshtein import distance


MODEL_TO_NAME = {
    'llama-3.1-8b-instruct': 'Meta-Llama-3.1-8B-Instruct',
    'gemma-2-9b-it': 'gemma-2-9b-it',
    'phi-3-medium-4k-instruct': 'Phi-3-medium-4k-instruct',
    'mistral-7b-instruct': 'Mistral-7B-Instruct-v0.3'
}

def load_data(task, model_type, model_name, quantization):
    root_dir = ''
    data_dir = ''

    target_path = f"{root_dir}/results/{model_name}/{quantization}/clean/{task}-{model_type}.json"
    data_file = json.load(open(target_path))

    if task == 'atis':
        data_file = list(filter(lambda entry: entry['label'] not in ['day_name', 'cheapest'],
                                data_file))

    texts = [entry['text'] for entry in data_file]
    model_outs = [entry['model_out'] for entry in data_file]

    intents = open(f"{data_dir}/{task}/intents.txt").read().splitlines()
    labels = torch.tensor([intents.index(entry['label']) for entry in data_file])
    preds = torch.tensor([intents.index(entry['model_out']) if entry['model_out'] in intents else -1 for entry in data_file])


    return labels, preds, texts, model_outs, intents

def map_to_intent(text, intents):
    distances = torch.tensor([distance(text, intent) for intent in intents])
    closest_match = intents[distances.argmin(dim=0)]
    return closest_match

def get_pred_from_text(model_output_clean, task, intents, do_map=False, do_print=True):
    failed_cleans = torch.zeros(len(model_output_clean))
    acc_intents = list(range(len(intents)))
    if task == 'atis':
        acc_intents = list(filter(lambda x: x not in [6, 8], acc_intents))

    preds = []
    labels = []
    for i_entry, entry in enumerate(model_output_clean):
        token_pred = entry['model_out']
        if do_map:
            token_pred = map_to_intent(token_pred, intents)
        label = intents.index(entry['label'])
        labels.append(label)

        if token_pred in intents:
            preds.append(intents.index(token_pred))
        else:
            failed_cleans[i_entry] = 1
            i_intent = acc_intents.index(label)
            pred = random.choice(acc_intents[:i_intent] + acc_intents[i_intent + 1:])
            preds.append(pred)

    if len(failed_cleans) > 0 and do_print:
        print(f"Failed to predict {len(failed_cleans.nonzero())}/{len(model_output_clean)} entries")

    preds = torch.tensor(preds)
    labels = torch.tensor(labels)
    return preds, labels, failed_cleans

def score(labels, preds, do_print=True, tag=None):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    if do_print:
        tag = f"[{tag}]" if tag is not None else ''
        print(f"{tag} acc: {(acc):.4f} macro-f1: {(f1):.4f}")
        print(f"{(acc * 100):.4g} {(f1 * 100):.4g}")
    return acc, f1

def pad_cfm(cfm, preds, labels, all_labels):
    labels_seen = list(set(preds.tolist() + labels.tolist()))
    for index in all_labels:
        if index not in labels_seen:
            l = cfm.shape[-1]
            cfm = np.c_[cfm[:, :index], np.zeros(l), cfm[:, index:]]
            cfm = np.r_[cfm[:index], np.zeros((1, l + 1)), cfm[index:]]
    return cfm

def cfm_and_plot(labels, preds, intents, normalize=None, annot=True, fmt='g', **kwargs):
    cfm = confusion_matrix(labels, preds, normalize=normalize)
    cfm = pad_cfm(cfm, preds, labels, list(range(len(intents))))
    sns.heatmap(cfm, xticklabels=intents, yticklabels=intents, fmt=fmt, annot=annot, **kwargs)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(pad=2)
    plt.show()

def normalise(v):
    return v / v.norm(dim=-1).unsqueeze(-1)
