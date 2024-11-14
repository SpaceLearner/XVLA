import os
import tensorflow_datasets as tfds
from scripts.generate_embodied_data.full_reasonings import generate_reasonings
import logging
import json

def main():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 设置环境变量
    
    # 设置数据集路径和名称
    data_dir = "/Users/gj/Documents/Projects/Embodied_Critic/data"
    dataset_name = "libero_spatial_no_noops"
    
    # 加载数据集构建器
    dataset_builder = tfds.builder(dataset_name, data_dir=data_dir)
    
    # # 获取完整数据集以验证步骤数
    ds = dataset_builder.as_dataset(split="train")
    # # 验证数据集中的步骤数
    # episode_steps = {}
    # for i in range(len(ds)):
    #     episode_id = i
    #     ds = dataset_builder.as_dataset(split=f"train[{episode_id}:{episode_id + 1}]")
    #     episode_steps[episode_id] = len(next(iter(ds))["steps"])

    
    # # 打印每个episode的步骤数
    # for episode_id, steps in episode_steps.items():
    #     logging.info(f"Episode {episode_id}: {steps} steps")
    
     # 设置输出路径
    output_dir = os.path.join(data_dir, "reasonings")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "reasonings.json")
    
    cur_episode = list(json.load(open(output_path, "r")).keys())
    cur_episode = max([int(i.strip().split("_")[-1]) for i in cur_episode])
    
    # 设置要处理的episode范围
    start_episode = cur_episode + 1
    end_episode = len(ds)
    episode_ids = range(start_episode, end_episode)
    
   
    
    try:
        # 生成reasonings
        generate_reasonings(
            builder=dataset_builder,
            episode_ids=episode_ids,
            save_path=output_path
        )
    except Exception as e:
        logging.error(f"生成reasonings时发生错误: {str(e)}")
        raise

def print_structure(obj, level=0):
    """递归打印嵌套数据结构"""
    indent = "  " * level
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(f"{indent}{key}:")
            print_structure(value, level + 1)
    elif hasattr(obj, 'keys'):
        for key in obj.keys():
            print(f"{indent}{key}:")
            print_structure(obj[key], level + 1)
    else:
        print(f"{indent}{type(obj)}")

if __name__ == "__main__":
    main() 