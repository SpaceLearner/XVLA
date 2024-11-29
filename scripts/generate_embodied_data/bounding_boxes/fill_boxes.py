import json

with open("./bboxes/full_bboxes.json", "r") as f:
    bboxes = json.load(f)

keys = list(bboxes.keys())
extracted_keys = []
for key in keys:
    extracted_keys += list(bboxes[key].keys())

expected_keys = {f"{i}" for i in range(432)}

missing_keys = expected_keys - set(extracted_keys)
print(len(missing_keys))
