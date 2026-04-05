import json

def get_unique_keys(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    unique_keys = set(data.keys())

    return unique_keys

print(len(get_unique_keys('/home/amondal/Codes/omnicount/outputs_sota/animals/animal_pred_15.json')))

print(get_unique_keys('/home/amondal/Codes/omnicount/outputs_sota/animals/animal_pred_15.json'))