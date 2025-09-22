import json

from string import Template
from tqdm import tqdm
from transformers import set_seed

from engine import GenericAdapter

set_seed(1234567890)

model = 'gemma-2-9b-it'
n_paraphrases = 3
model_path = ''
quantization = '4bit'

ROOT_DIR = ''
DATA_DIR = f"{ROOT_DIR}/data"
OUTPUT_DIR = f"{ROOT_DIR}/desc_paraphrases"

tasks = ['atis', 'snips', 'clinic150', 'massive']

prompt_template = Template("""
You are part of a task-oriented dialogue system designed to classify a given user utterance to an intent from a set of intents.
Your task is: given a description of an intent, generate a paraphrase of the description as an alternate description fitting the intent.

Given $intent with the description '$description'.
Generate a description for the above intent, do not provide anything else.
""")

adapter = GenericAdapter(model_path=model_path, quantization=quantization)

for task in tasks:
    intents = open(f"{DATA_DIR}/{task}/intents.txt").read().splitlines()
    descriptions = json.load(open(f"{DATA_DIR}/{task}/descriptions.json"))
    para_descs = {}
    for intent in tqdm(intents):
        para_descs[intent] = []
        params = {
            'intent': intent,
            'description': descriptions[intent]
        }
        prompt = prompt_template.substitute(**params).strip()
        prompt = adapter.format_prompt(prompt)
        for _ in range(n_paraphrases):
            out, _ = adapter.complete(prompt)
            para_descs[intent].append(out.strip())
    with open(f"{OUTPUT_DIR}/{task}-{model}-desc.json", 'w') as output_file:
        output_file.write(json.dumps(para_descs, indent=4))
