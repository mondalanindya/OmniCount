import json

from paths import OUTPUTS_ROOT

def get_unique_keys(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    unique_keys = set(data.keys())

    return unique_keys

json_file_path = OUTPUTS_ROOT / "animals" / "animal_pred_15.json"

print(len(get_unique_keys(json_file_path)))

print(get_unique_keys(json_file_path))