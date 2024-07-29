import json
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base_folder", type=str, default="outputs/whisper-base-lower_text_label_aligned_v03", help="Path to base folder")
args = parser.parse_args()

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


data = read_json(f"{args.base_folder}/generation_config.json")
data["suppress_tokens"] = []
write_json(data, f"{args.base_folder}/generation_config.json")