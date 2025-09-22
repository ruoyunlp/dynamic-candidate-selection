import os
import json

from argparse import ArgumentParser
from pathlib import Path

def json_to_jsonl(path):
    content = json.load(open(path))
    lines = ''.join([json.dumps(entry) + '\n' for entry in content])
    return lines


def jsonl_to_json(path):
    lines = open(path).read().splitlines()
    content = [json.loads(entry) for entry in lines]
    return json.dumps(content, indent=4)


def jsonifier(path, to='json'):
    # converts a given file at 'path' to target filetype
    assert os.path.exists(path), f"File not found at {repr(path)}"
    path = Path(path)

    text = None
    output_path = os.path.join(os.path.dirname(path), path.stem + '.' + to)
    if to == 'json':
        assert path.suffix == '.jsonl'
        text = jsonl_to_json(path)
    elif to == 'jsonl':
        assert path.suffix == '.json'
        text = json_to_jsonl(path)
    else:
        raise ValueError(f"Unsupported processing target file type '{to}'")

    with open(output_path, 'w') as output_file:
        output_file.write(text)

    print(f"Successfully converted {path.name} to {to}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('-to', default='json', required=False)

    args = parser.parse_args()
    jsonifier(**args.__dict__)
