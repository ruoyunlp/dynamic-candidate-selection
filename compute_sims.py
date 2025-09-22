# ===================================================
### CONTAINS UTILITY FUNCTIONS RELATING TO CLASS SIMS
# ===================================================

import json
import torch

from sklearn.metrics import accuracy_score, f1_score

root_dir = '.'
model_type = 'bge-large-en-v1.5'
# model_type = 'gte-large'
# combinations = 'e+p+om'
combinations = 'e'

def normalise(v):
    return v / v.norm(dim=-1).unsqueeze(-1)

def score(labels, preds, do_print=True, tag=None):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    if do_print:
        tag = f"[{tag}]" if tag is not None else ''
        print(f"{tag} acc: {(acc):.4f} macro-f1: {(f1):.4f}")
        print(f"{(acc * 100):.4g} {(f1 * 100):.4g}")
    return acc, f1

def load_task(task):
    data = json.load(open(f"{root_dir}/masked-embed2/{task}-masked.json"))
    if task == 'atis':
        data = list(filter(lambda x: x['intent'] not in ['cheapest', 'day_name'], data))
    intents = open(f"{root_dir}/data/{task}/intents.txt").read().splitlines()
    descriptions = json.load(open(f"{root_dir}/data/{task}/descriptions.json"))
    labels = torch.tensor([intents.index(entry['intent']) for entry in data])

    return data, intents, descriptions, labels

def find_candidates(task, top_k, intents):
    sims = torch.load(f"sims/{task}-{model_type}-sims-{combinations}.pt", weights_only=True)
    sims_sorted_indices = torch.sort(sims, dim=1, descending=True).indices
    top_k_candidates = sims_sorted_indices[:, :top_k]
    outputs = []
    for i in range(len(top_k_candidates)):
        outputs.append([intents[cand] for cand in top_k_candidates[i]])
    return outputs

def cache_candidates(task, top_k, intents):
    import os
    candidates = find_candidates(task, top_k, intents)
    cand_dir = f"."
    if not os.path.exists(cand_dir):
        os.mkdir(cand_dir)
    with open(f"{cand_dir}/{task}-{model_type}-{combinations}-{top_k}-cands.jsonl", 'w') as output_file:
        output_file.write(json.dumps(candidates))


# ===================================================
### SHOULD BE ONE-TIME USE FOR COMPUTING SIM MATRICES
# ===================================================
if __name__ == "__main__":
    tasks = ['atis', 'snips', 'clinic150', 'massive']
    for task in tasks:
        data_dir = f"{root_dir}/data/{task}"

        data = json.load(open(f"data/{task}.json"))
        intents = open(f"{root_dir}/data/{task}/intents.txt").read().splitlines()
        labels = torch.tensor([intents.index(entry['intent']) for entry in data])

        embeds = torch.load(f"{root_dir}/output-unnorm/{task}-{model_type}-descriptions-embeds.pt", weights_only=True)
        desc_embeds = torch.load(f"{root_dir}/output-unnorm/{task}-{model_type}-descriptions-desc-embeds.pt", weights_only=True)
        masked_embeds = torch.load(f"{root_dir}/output-unnorm/{task}-{model_type}-masked-embeds.pt", weights_only=True)
        pp_embeds = torch.load(f"{root_dir}/userdesc/stablelm-2-1_6b-chat/embeds/{task}-{model_type}-descriptions-embeds.pt", weights_only=True)

        choice_k = 3
        overlaps = torch.load(f"{root_dir}/masked-embed2/analysis/overlaps/{task}-{model_type}-{choice_k}-choices.pt", weights_only=True)

        if task == 'atis':
            filtered_idx = (labels != 6) & (labels != 8)
            embeds = embeds[filtered_idx]
            labels = labels[filtered_idx]
            masked_embeds = masked_embeds[filtered_idx]
            pp_embeds = pp_embeds[filtered_idx]
            overlaps = torch.ones(overlaps.shape)

        sims = embeds + pp_embeds + overlaps.unsqueeze(-1) * masked_embeds
        # sims = embeds

        sims = normalise(sims) @ normalise(desc_embeds).T
        if task == 'atis':
            sims[:, [6, 8]] = -1
        preds = torch.argmax(sims, dim=1)
        print(score(labels, preds))  # SANITY CHECK

        torch.save(sims, f"sims/{task}-{model_type}-sims-{combinations}.pt")