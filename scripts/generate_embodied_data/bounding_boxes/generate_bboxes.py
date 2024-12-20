import argparse
import json
import os
import time
import warnings

import tensorflow_datasets as tfds
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from utils import NumpyFloatValuesEncoder, post_process_caption

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()

parser.add_argument("--id", type=int)
parser.add_argument("--gpu", type=int)
parser.add_argument("--splits", default=24)
parser.add_argument("--data-path", type=str)
parser.add_argument("--result-path", default="./bboxes")

args = parser.parse_args()
bbox_json_path = os.path.join(args.result_path, f"results_{args.id}_bboxes.json")

print("Loading data...")
split_percents = 100 // args.splits
start = args.id * split_percents
end = (args.id + 1) * split_percents

# ds = tfds.load("bridge_orig", data_dir=args.data_path, split=f"train[{start}%:{end}%]")

<<<<<<< HEAD
dataset_builder = tfds.builder("libero_spatial_reasoning", data_dir="/Users/gj/Documents/Projects/Embodied_Critic/data")
=======
dataset_builder = tfds.builder("libero_spatial", data_dir="/home/admin/workspace/data")
>>>>>>> refs/remotes/origin/main
ds = dataset_builder.as_dataset(split=f"train[{start}%:{end}%]", shuffle_files=False)
print("Done.")

print("Loading Prismatic descriptions...")
results_json_path = "./descriptions/full_descriptions.json"
with open(results_json_path, "r") as f:
    results_json = json.load(f)
print("Done.")

print(f"Loading gDINO to device {args.gpu}...")
model_id = "IDEA-Research/grounding-dino-base"
device = f"cuda:{args.gpu}"

processor = AutoProcessor.from_pretrained(model_id, size={"shortest_edge": 256, "longest_edge": 256})
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
print("Done.")

BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.3

bbox_results_json = {}
for ep_idx, episode in enumerate(ds):

    episode_id = ep_idx + args.id * 18 # episode["episode_metadata"]["episode_id"].numpy()
    file_path  = episode["episode_metadata"]["file_path"].numpy().decode()
    print(f"ID {args.id} starting ep: {episode_id}, {file_path}")

    if file_path not in bbox_results_json.keys():
        bbox_results_json[file_path] = {}

    # episode_json = results_json[file_path][str(episode_id)]
    description = "wooden cabinet, black bowl, plate, packet, drawer" # episode_json["caption"]

    start = time.time()
    bboxes_list = []
    for step_idx, step in enumerate(episode["steps"]):
        if step_idx == 0:
            lang_instruction = step["language_instruction"].numpy().decode()
        image = Image.fromarray(step["observation"]["image"].numpy())
        inputs = processor(
            images=image,
            text=post_process_caption(description, lang_instruction),
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]],
        )[0]

        logits, phrases, boxes = (
            results["scores"].cpu().numpy(),
            results["labels"],
            results["boxes"].cpu().numpy(),
        )

        bboxes = []
        for lg, p, b in zip(logits, phrases, boxes):
            b = list(b.astype(int))
            lg = round(lg, 5)
            bboxes.append((lg, p, b))
            break

        bboxes_list.append(bboxes)
        # break
    end = time.time()
    bbox_results_json[file_path][str(ep_idx)] = {
        "episode_id": int(episode_id),
        "file_path": file_path,
        "bboxes": bboxes_list,
    }

    with open(bbox_json_path, "w") as f:
        json.dump(bbox_results_json, f, cls=NumpyFloatValuesEncoder)
    print(f"ID {args.id} finished ep ({ep_idx} / {len(ds)}). Elapsed time: {round(end - start, 2)}")
