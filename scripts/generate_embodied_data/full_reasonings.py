import json
import os
import re
import time
import base64
from PIL import Image
import numpy as np
from collections import defaultdict
import cv2

from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from scripts.generate_embodied_data.primitive_movements import get_move_primitives_episode

def encode_image_to_base64(image_array):
    """将numpy数组转换为base64字符串"""
    image = Image.fromarray(image_array)
    import io
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


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
                    base64_image = encode_image_to_base64(image)
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

def find_task_occurrences(input_string, tags):
    """
    匹配每一步的reasoning，允许标签之间有空格
    """
    
    # 首先尝试提取所有的步骤
    step_pattern = r'(\d+):\s*"(.*?)"(?=,|\s*})'
    steps = re.findall(step_pattern, input_string, re.DOTALL)
    
    print(f"\nFound {len(steps)} potential steps")
    if steps:
        print("First step:", steps[0][0], "content preview:", steps[0][1][:100])
    
    cleaned_matches = []
    for step_num, step_content in steps:
        # 为每个标签构建单独的模式
        step_data = [step_num]
        for tag in tags:
            pattern = f'<{tag}>(.*?)</{tag}>'
            match = re.search(pattern, step_content, re.DOTALL)
            if match:
                step_data.append(match.group(1).strip())
            else:
                print(f"\nMissing tag {tag} in step {step_num}")
                print("Step content:", step_content)
                break
        
        if len(step_data) == len(tags) + 1:  # +1 for step number
            cleaned_matches.append(tuple(step_data))
    
    print(f"\nSuccessfully matched {len(cleaned_matches)} complete steps")
    if cleaned_matches:
        print("First complete match:", cleaned_matches[0])
    
    return cleaned_matches


def extract_reasoning_dict(reasoning_output, tags=("task", "plan", "subtask", "subtask_reason", "move", "move_reason")):
    if reasoning_output is None:
        return dict()

    trajectory = dict()

    try:
        # 清理输入字符串，移除可能的Python字典格式
        cleaned_output = reasoning_output
        if "FINISHED" in cleaned_output:
            cleaned_output = cleaned_output.split("FINISHED")[0]
        
        # 尝试找到字典的开始和结束
        dict_start = cleaned_output.find("{")
        dict_end = cleaned_output.rfind("}")
        if dict_start != -1 and dict_end != -1:
            cleaned_output = cleaned_output[dict_start:dict_end+1]
        
        print("\nCleaned output preview:")
        print(cleaned_output)
        
        matches = find_task_occurrences(cleaned_output, tags)
        
        # 如果没有找到匹配项，返回None以触发重试
        if not matches:
            print("No matches found, returning None to trigger retry")
            return None
            
        for match in matches:
            step_num = int(match[0])
            trajectory[step_num] = dict(zip(tags, match[1:]))
            
    except Exception as e:
        print(f"Error extracting reasoning: {str(e)}")
        print(f"Reasoning output: {reasoning_output}")
        return None
        
    return trajectory[list(sorted(trajectory.keys(), key=int))[0]]


def get_reasoning_dict(features, metadata, lm, images):
    """基于轨迹分析生成reasoning"""
    language_instruction = metadata["language_instruction"]
    total_steps = len(features["move_primitive"])
    episode_id = metadata["episode_id"]
    
    # 分析轨迹模式
    action_groups = analyze_trajectory(features)
    print(f"Found {len(action_groups)} action groups")
    
    # 存储所有步骤的reasoning
    all_reasonings = {}
    group_subtasks = []  # 存储每个组的subtask
    
    # 为每个组单独生成reasoning
    for group_idx, group in enumerate(action_groups):
        print(f"\nProcessing group {group_idx + 1}/{len(action_groups)}:")
        print(f"Step range: {group['start_idx']}-{group['end_idx']}")
        print(f"Action type: {group['move_type']}")
        
        # 获取该组的起始和结束图片
        start_image = images[group['start_idx']]
        end_image = images[group['end_idx']]
        
        # 构建之前组的reasoning描述
        previous_groups = ""
        for prev_idx in range(group_idx):
            prev_group = action_groups[prev_idx]
            prev_reasoning = all_reasonings.get(prev_group['start_idx'])
            if prev_reasoning:
                previous_groups += f"""Group {prev_idx + 1}:
- Steps {prev_group['start_idx']}-{prev_group['end_idx']}
- Action: {prev_group['move_type']}
- Subtask: {prev_reasoning.get('subtask', 'Unknown')}
- Reasoning: {prev_reasoning.get('subtask_reason', 'Unknown')}

"""
        
        # 构建该组的prompt
        prompt = f"""# Generate reasoning for robot actions

Task instruction: "{language_instruction}"

{previous_groups if previous_groups else ""}## Current Action Group (Group {group_idx + 1}/{len(action_groups)})
- Step range: {group['start_idx']} to {group['end_idx']}
- Action type: {group['move_type']}
- Duration: {group['duration']} steps
- State change: {[round(x, 3) for x in group['state_change']]}

The first image shows the state at the start of this action group.
The second image shows the state at the end of this action group.

## Required Output
Generate reasoning for the action group with the following tags:
<task>Remaining task description</task>
<plan>List of remaining high-level steps</plan>
<subtask>Current high-level step</subtask>
<subtask_reason>Why execute this step now, considering previous actions</subtask_reason>
<move>{group['move_type']}</move>
<move_reason>Why execute this movement</move_reason>

There is no need to repeat a subtask many times. Analyze the start and end state of the action group and analyze what remains to do and then generate the reasoning for the action group in the following format, with the first step of the action group:
{{
    {group['start_idx']}: "<task>...</task><plan>...</plan><subtask>...</subtask><subtask_reason>...</subtask_reason><move>...</move><move_reason>...</move_reason>",
    ...
}}

FINISHED"""
        
        # 最多尝试3次生成reasoning
        max_attempts = 3
        group_reasoning = None
        
        for attempt in range(max_attempts):
            # 生成该组的reasoning
            reasoning_output = lm.generate(prompt, [start_image, end_image])
            group_reasoning = extract_reasoning_dict(reasoning_output)
            
            if group_reasoning is not None:
                print(f"Successfully generated reasoning on attempt {attempt + 1}")
                break
            else:
                print(f"Failed to generate valid reasoning on attempt {attempt + 1}, retrying...")
        
        if group_reasoning is None:
            raise ValueError(f"Failed to generate valid reasoning for group {group_idx} after {max_attempts} attempts")
        
        group_subtasks.append(group_reasoning.get('subtask', 'Unknown'))
        all_reasonings.update({group['start_idx']: group_reasoning})
    
    # 生成整体plan
    # 构建action sequence字符串
    action_sequence = ""
    for idx, (group, subtask) in enumerate(zip(action_groups, group_subtasks)):
        action_sequence += f"""Group {idx + 1}:
- Steps {group['start_idx']}-{group['end_idx']}
- Action: {group['move_type']}
- Subtask: {subtask}
"""
    
    full_sequence_prompt = f"""# Summarize the action sequence

Task instruction: "{language_instruction}"

## Action Groups Sequence
{action_sequence}

## Required Output
Based on the sequence of actions and their subtasks, provide a ONE-SENTENCE summary that describes how these subtasks work together to accomplish the main task.
Focus on the high-level flow of actions, not the details.

Example format: "1. [first action], 2. [second action], ... 3. [last action]."
"""
    
    # 生成整体plan
    full_plan = lm.generate(full_sequence_prompt, [images[0], images[-1]])
    print("\nOverall plan generated:", full_plan)
    
    # 使用整体plan更新每个组的reasoning
    all_reasonings = propagate_reasoning(all_reasonings, action_groups)
    
    # 更新每个步骤的plan
    for step in all_reasonings:
        all_reasonings[step]['plan'] = full_plan
            
    # 最终验证
    assert len(all_reasonings) == total_steps
    
    return all_reasonings

def convert_to_serializable(obj):
    """将对象转换为可JSON序列化的格式"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def build_single_reasoning(episode_id, builder, lm):
    ds = builder.as_dataset(split=f"train[{episode_id}:{episode_id + 1}]")
    
    try:
        episode = next(iter(ds))
    except Exception as e:
        print(f"Error getting episode from dataset: {str(e)}")
        raise

    # 获取steps
    steps = list(episode["steps"])
    # print(episode["episode_metadata"])
    if not steps:
        raise ValueError("No steps found in episode")

    total_steps = len(steps)
    
    ft = dict()
    # 确保转换为Python原生类型
    ft["state_3d"] = [convert_to_serializable(step["observation"]["state"][:3].numpy()) for step in steps]
    move_primitives = get_move_primitives_episode(episode)
    ft["move_primitive"] = [move[0] for move in move_primitives]
    
    # 验证特征数量
    assert len(ft["state_3d"]) == total_steps, f"state_3d长度 ({len(ft['state_3d'])}) 与步骤数 ({total_steps}) 不匹配"
    assert len(ft["move_primitive"]) == total_steps, f"move_primitive长度 ({len(ft['move_primitive'])}) 与步骤数 ({total_steps}) 不匹配"

    # import pdb; pdb.set_trace()

    # 获取所有图片
    images = [step["observation"]["image"].numpy() for step in steps]
    language_instruction = str(steps[0]["language_instruction"].numpy().decode())

    mt = {
        "episode_id": str(episode_id),
        "file_path": "episode_" + str(episode_id),  # episode["episode_metadata"]["file_path"].numpy().decode("utf-8"),
        "n_steps": total_steps,
        "language_instruction": language_instruction,
    }

    # 使用完整的图片序列生成reasoning
    reasoning = get_reasoning_dict(ft, mt, lm, images)
    
    # 验证reasoning数量
    assert len(reasoning) == total_steps, \
        f"Episode {episode_id}: reasoning数量 ({len(reasoning)}) 与步骤数 ({total_steps}) 不匹配"
    
    # 验证reasoning的步骤编号
    assert set(reasoning.keys()) == set(range(total_steps)), \
        f"Episode {episode_id}: reasoning的步骤编号与预期不匹配。缺失步骤: {set(range(total_steps)) - set(reasoning.keys())}"
    
    entry = {"reasoning": reasoning, "features": ft, "metadata": mt}
    return convert_to_serializable(entry)


def generate_reasonings(builder, episode_ids, save_path="reasonings.json"):
    reasonings = dict()
    lm = GPT4()

    if os.path.exists(save_path):
        print(save_path, "existing, loading contents")
        with open(save_path, "r") as f:
            reasonings = json.load(f)
        print("loaded reasonings:", sum([len(v) for v in reasonings.values()]), "entries")

    for i in episode_ids:
        try:
            entry = build_single_reasoning(i, builder, lm)
            
            file_path  = entry["metadata"]["file_path"]
            episode_id = entry["metadata"]["episode_id"]

            if file_path not in reasonings:
                reasonings[file_path] = {}
            
            reasonings[file_path][episode_id] = entry

            if (i + 1) % 1 == 0:
                print(f"Saving intermediate results after episode {i}")
                with open(save_path, "w") as out_f:
                    json.dump(reasonings, out_f)

        except Exception as e:
            print(f"Error processing episode {i}: {str(e)}")
            with open(save_path, "w") as out_f:
                json.dump(convert_to_serializable(reasonings), out_f)  # 确保错误时也能保存
            continue

    # 最终保存
    with open(save_path, "w") as out_f:
        json.dump(convert_to_serializable(reasonings), out_f)


def analyze_trajectory(features):
    """分析轨迹，将相似的动作分组并识别模式"""
    moves = features["move_primitive"]
    states = features["state_3d"]
    
    # 初始化分组
    action_groups = []
    current_group = {
        "start_idx": 0,
        "end_idx": 0,
        "moves": [moves[0]],
        "states": [states[0]],
        "move_type": moves[0]
    }
    
    # 根据连续相同作进行分组
    for i in range(1, len(moves)):
        if moves[i] == current_group["move_type"]:
            current_group["moves"].append(moves[i])
            current_group["states"].append(states[i])
            current_group["end_idx"] = i
        else:
            action_groups.append(current_group)
            current_group = {
                "start_idx": i,
                "end_idx": i,
                "moves": [moves[i]],
                "states": [states[i]],
                "move_type": moves[i]
            }
    
    # 添加最后一组
    action_groups.append(current_group)
    
    # 分析每组的特征
    for group in action_groups:
        # 计算状态变化
        start_state = np.array(group["states"][0])
        end_state = np.array(group["states"][-1])
        state_change = end_state - start_state
        
        group.update({
            "duration": len(group["moves"]),
            "state_change": state_change.tolist(),
            "step_indices": list(range(group["start_idx"], group["end_idx"] + 1))
        })
    
    return action_groups

def propagate_reasoning(raw_reasonings, action_groups):
    """将reasoning传播到所有步骤，确保每个步骤都有对应的reasoning"""
    complete_reasonings = {}
    
    # 计算总骤数
    total_steps = max(group["end_idx"] for group in action_groups) + 1
    
    # 遍历每个动作组
    for group in action_groups:
        start_idx = group["start_idx"]
        end_idx   = group["end_idx"]
        
        # 获取该组的模板reasoning
        template = None
        # 首先尝试使用组内第一步的reasoning作为模板
        if start_idx in raw_reasonings:
            template = raw_reasonings[start_idx]
        else:
            # 如果组内第一步没有reasoning，查找组内任何一个有reasoning的步骤
            for step in range(start_idx, end_idx + 1):
                if step in raw_reasonings:
                    template = raw_reasonings[step]
                    break
        
        if template is None:
            raise ValueError(f"动作组 {start_idx}-{end_idx} 没有找到可用的reasoning模板")
            
        # 为组内的每一步设置reasoning
        for step in range(start_idx, end_idx + 1):
            if step in raw_reasonings:
                # 如果该步骤已有reasoning，使用原有的
                complete_reasonings[step] = raw_reasonings[step]
            else:
                # 否则使用模板
                complete_reasonings[step] = template.copy()
    
    # 验证是否所有步骤都有reasoning
    expected_steps = set(range(total_steps))
    actual_steps = set(complete_reasonings.keys())
    
    # 验证步骤完整性
    assert expected_steps == actual_steps, \
        f"步骤不完整。缺失步骤: {expected_steps - actual_steps}, 多余步骤: {actual_steps - expected_steps}"
    
    # 验证步骤连续性
    assert len(complete_reasonings) == total_steps, \
        f"reasoning数量 ({len(complete_reasonings)}) 与总步骤数 ({total_steps}) 不匹配"
    
    return complete_reasonings

if __name__ == "__main__":
    client = GPT4()
    client.generate("hello", None)