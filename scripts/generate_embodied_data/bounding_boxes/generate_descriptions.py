import argparse
import json
import os
import warnings
import tensorflow_datasets as tfds
from PIL import Image
from tqdm import tqdm
from utils import NumpyFloatValuesEncoder
from openai import AzureOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt
import base64
from io import BytesIO

class GPT4:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

    @retry(wait=wait_exponential(min=1, max=5), stop=stop_after_attempt(6))
    def generate(self, prompt, images=None):
        try:
            messages = [{"role": "user", "content": []}]
            
            # 如果有图片，添加所有图片
            if images is not None:
                for image in images:
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    })
            
            # 添加文本提示
            messages[0]["content"].append({
                "type": "text",
                "text": prompt
            })

            response = self.client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                messages=messages,
                temperature=0.1,
                max_tokens=10000,
            )
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error during API call: {e}")
            raise

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int)
parser.add_argument("--splits", default=4, type=int)
parser.add_argument("--results-path", default="./")

args = parser.parse_args()

warnings.filterwarnings("ignore")

split_percents = 100 // args.splits
start = args.id * split_percents
end = (args.id + 1) * split_percents

# Load Bridge V2
dataset_builder = tfds.builder("libero_spatial_reasoning", data_dir="/home/admin/workspace/data/")
ds = dataset_builder.as_dataset(split=f"train[{start}%:{end}%]")
print("Done.")

# 初始化GPT4客户端
lm = GPT4()

results_json_path = os.path.join(args.results_path, f"results_{args.id}.json")

def create_user_prompt(lang_instruction):
    user_prompt = "Briefly describe the things in this scene and their spatial relations to each other."
    lang_instruction = lang_instruction.strip()
    if len(lang_instruction) > 0 and lang_instruction[-1] == ".":
        lang_instruction = lang_instruction[:-1]
    if len(lang_instruction) > 0 and " " in lang_instruction:
        user_prompt = f"The robot task is: '{lang_instruction}.' " + user_prompt
    return user_prompt

results_json = {}
for episode in tqdm(ds):
    episode_id = episode["episode_metadata"]["episode_id"].numpy()
    file_path = episode["episode_metadata"]["file_path"].numpy().decode()
    for step in episode["steps"]:
        lang_instruction = step["language_instruction"].numpy().decode()
        image = Image.fromarray(step["observation"]["image_0"].numpy())

        user_prompt = create_user_prompt(lang_instruction)
        caption = lm.generate(user_prompt, [image])
        break

    episode_json = {
        "episode_id": int(episode_id.strip().split("_")[-1]),
        "file_path": file_path,
        "caption": caption,
    }

    if file_path not in results_json.keys():
        results_json[file_path] = {}

    results_json[file_path][int(episode_id)] = episode_json

    with open(results_json_path, "w") as f:
        json.dump(results_json, f, cls=NumpyFloatValuesEncoder)
