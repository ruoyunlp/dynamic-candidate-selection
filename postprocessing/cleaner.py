import re
import os
import sys
import json
import torch
import random

from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score
from Levenshtein import distance
from typing import List
from transformers import AutoModel, AutoTokenizer

random.seed(1234567890)

def score(labels, preds, do_print=True, tag=None):
    acc = accuracy_score(labels, preds) * 100
    f1 = f1_score(labels, preds, average='macro') * 100
    if do_print:
        tag = f"[{tag}]" if tag is not None else ''
        print(f"{tag} acc: {(acc):.4f} macro-f1: {(f1):.4f}")
    return acc, f1

def masked_mean(tensor, mask):
    """ Function for calculating mean of embeddings with masking

    Args:
        tensor (torch.Tensor): batch_size * seq_len * hidden_size
        mask (torch.Tensor): batch_size * seq_len

    Returns:
        _type_: _description_
    """
    hidden_size = tensor.shape[-1]

    # set masked tokens (i.e. [PAD]) to 0
    out = tensor.reshape(-1, hidden_size) * mask.flatten().reshape(-1, 1)

    # reset shape to batch_size x seq_len x hidden_size
    out = out.reshape(tensor.shape)

    # sum and divide to calculate mean
    out = out.sum(dim=1) / mask.sum(dim=1).reshape(-1, 1)
    return out


class BaseInferenceModel:
    def get_sentence_embeds(self, sents: List[str], norm=True) -> torch.Tensor:
        """ Produce contextual representations for given sentences

        Args:
            sents (List[str]): List of sentences to get representations for
            norm (bool, optional): normalise embeddings. Defaults to True.

        Returns:
            torch.Tensor: contextual sentence representations, torch Tensor of shape num_sentences x hidden_size
        """
        raise NotImplementedError()

class GTEModel(BaseInferenceModel):
    def __init__(self, model_type, device) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_type).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.device = device

    def get_sentence_embeds(self, sents: List[str], norm=True) -> torch.Tensor:
        encodings = self.tokenizer(sents, padding=True, truncation=True, return_tensors='pt')
        encodings = encodings.to(self.device)
        input_ids = encodings.input_ids
        attention_mask = encodings.attention_mask

        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = out.last_hidden_state.detach()

        # omit [MASK] token embeddings from averaging
        attention_mask = attention_mask & (input_ids != self.tokenizer.mask_token_id)
        embeds = masked_mean(last_hidden_state, attention_mask).cpu()

        if norm:
            embeds = embeds / embeds.norm(dim=-1).unsqueeze(-1)

        return embeds

model_to_name = {
    'llama-3.1-8b-instruct': 'Meta-Llama-3.1-8B-Instruct',
    'gemma-2-9b-it': 'gemma-2-9b-it',
    'phi-3-medium-4k-instruct': 'Phi-3-medium-4k-instruct',
    'mistral-7b-instruct': 'Mistral-7B-Instruct-v0.3'
}

all_tasks = ['atis', 'snips', 'clinic150', 'massive']
root_dir = '.'

def map_to_intent(text, intents):
    distances = torch.tensor([distance(text, intent) for intent in intents])
    closest_match = intents[distances.argmin(dim=0)]
    return closest_match

def enc_map_to_intent(text, intents, descriptions, model):
    descs = [descriptions[intent] for intent in intents]
    embeds_text = model.get_sentence_embeds([text])
    embeds_desc = model.get_sentence_embeds(descs)
    assert len(embeds_text.shape) == 2
    assert embeds_text.shape[1] == embeds_desc.shape[1]
    sims = embeds_text @ embeds_desc.T
    pred = intents[sims.argmax(dim=-1)]
    return pred

def clean_model_out(text, intents, do_print=False):
    try:
        out = text
        mapper = {
            "flights": "flight"
        }

        if text in mapper:
            out = mapper[text]

        out = re.findall(r'\S+', out)[-1]
        assert out in intents
    except Exception as e:
        # if do_print
        #     print(f"Unable to clean output: {repr(text)}, found: {repr(out)}")
        pass

    return out

def clean(args, quantization=None, model_type=None, do_print=False, candidates=None):
    do_eval = False
    do_clean = False
    show_failed = False

    model_name = model_to_name[model_type]

    tasks = ['atis']
    if len(args) > 1:
        tasks = list(filter(lambda x: x in args, all_tasks))
        do_clean = 'clean' in args
        do_eval = 'eval' in args
        use_enc = 'enc' in args

        if use_enc:
            encoder_model = GTEModel("bge-large-en-v1.5", 'cuda')

    for task in tasks:
        print(f"Task: {task.upper()}")
        intents = open(f"{root_dir}/data/{task}/intents.txt").read().splitlines()
        output_raw_path = f"{model_name}/{quantization}/raw/{task}-{model_type}.json"
        output_clean_path = f"{model_name}/{quantization}/clean"
        if not os.path.exists(output_clean_path): os.mkdir(output_clean_path)
        output_clean_path = f"{model_name}/{quantization}/clean/{task}-{model_type}.json"

        if not os.path.exists(output_raw_path):
            print(f"Failed to find file {output_raw_path}")
            continue

        shortlists = candidates

        data = json.load(open(f"{root_dir}/data/{task}/data-full-shuffled.json"))['data']
        model_output_raw = json.load(open(output_raw_path))
        print(f"Found {len(model_output_raw)} raw results")

        if shortlists is None:
            shortlists = [intents for _ in range(len(data))]

        if do_clean:
            model_output_clean = []
            for i, (model_out, entry, shortlist) in enumerate(zip(model_output_raw, data, shortlists)):
                out = {
                    'index': entry['index'],
                    **model_out
                }
                out['model_out'] = clean_model_out(model_out['model_out'], intents, do_print=do_print)
                model_output_clean.append(out)

            with open(output_clean_path, 'w') as output_file:
                output_file.write(json.dumps(model_output_clean, indent=4))

        if do_eval:
            target_data_path = output_raw_path
            if do_clean and os.path.exists(output_clean_path):
                target_data_path = output_clean_path

            if use_enc:
                descriptions = json.load(open(f"{root_dir}/data/{task}/descriptions.json"))

            target_model_output = json.load(open(target_data_path))
            failed_cleans = []
            acc_intents = list(range(len(intents)))
            if task == 'atis':
                acc_intents = list(filter(lambda x: x not in [6, 8], acc_intents))

            preds = []
            labels = []
            for entry, shortlist in zip(target_model_output, shortlists):
                token_pred = entry['model_out']
                label = intents.index(entry['label'])
                labels.append(label)

                if token_pred in intents:
                    preds.append(intents.index(token_pred))
                else:
                    failed_cleans.append(token_pred)
                    if use_enc:
                        pred = intents.index(enc_map_to_intent(token_pred, shortlist, descriptions, encoder_model))
                    else:
                        pred = intents.index(map_to_intent(token_pred, shortlist))
                    preds.append(pred)

            if len(failed_cleans) > 0:
                print(f"Failed to predict {len(failed_cleans)}/{len(target_model_output)} entries")

            scores = score(labels, preds)
            if len(tasks) == 1: return scores


if __name__ == "__main__":
    random.seed(1234567890)
    [quantization, model_type, *args] = sys.argv[1:]
    clean(args, quantization, model_type)
