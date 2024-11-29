import os
import tensorflow_datasets as tfds
import logging
import json
import argparse
from primitive_movements import get_move_primitives_episode

def generate_primitive_dataset(builder, episode_ids, save_path):
    """从数据集中提取primitive动作，并按照文件路径组织"""
    # 如果文件已存在，加载现有数据
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            dataset = json.load(f)
    else:
        dataset = {}
    
    for episode_id in episode_ids:
        try:
            # 获取episode数据
            ds = builder.as_dataset(
                split=f"train[{episode_id}:{episode_id + 1}]",
                shuffle_files=False
            )
            episode_data = next(iter(ds))
            
            # 获取文件路径和episode_id
            file_path = episode_data["episode_metadata"]["file_path"].numpy().decode("utf-8")
            episode_id_str = episode_data["episode_metadata"]["episode_id"].numpy().decode("utf-8")
            
            # 获取language instruction
            language_instruction = next(iter(episode_data["steps"]))["language_instruction"].numpy().decode("utf-8")
            
            # 获取primitive moves
            primitives = get_move_primitives_episode(episode_data)
            moves = [move[0] for move in primitives]
            
            # 组织数据结构
            if file_path not in dataset:
                dataset[file_path] = {}
                
            dataset[file_path][episode_id_str] = {
                "metadata": {
                    "file_path": file_path,
                    "episode_id": episode_id_str,
                    "language_instruction": language_instruction,
                    "n_steps": len(episode_data["steps"])
                },
                "features": {
                    "move_primitive": moves,
                    "bboxes": {},
                    "gripper_position": {}
                },
                "reasoning": []
            }
            
            # 定期保存到文件
            if episode_id % 10 == 0:
                with open(save_path, 'w') as f:
                    json.dump(dataset, f, indent=2)
                logging.info(f"处理完成episode {episode_id}")
                
        except Exception as e:
            logging.error(f"处理episode {episode_id}时出错: {str(e)}")
            continue
    
    # 最后保存一次
    with open(save_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    return dataset

def parse_args():
    parser = argparse.ArgumentParser(description='生成primitive moves数据集')
    parser.add_argument('--data_dir', type=str, default="/home/admin/workspace/data",
                        help='数据集根目录路径')
    parser.add_argument('--dataset_name', type=str, default="libero_spatial_reasoning",
                        help='数据集名称')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 加载数据集构建器
    dataset_builder = tfds.builder(args.dataset_name, data_dir=args.data_dir)
    
    # 获取数据集
    ds = dataset_builder.as_dataset(split="train", shuffle_files=False)
    
    # 设置输出路径
    output_dir = os.path.join(args.data_dir, args.dataset_name, "primitives")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "primitives.json")
    
    # 如果文件存在，获取当前处理到的episode
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            existing_data = json.load(f)
        # 找到最大的episode编号
        max_episode = 0
        for file_path in existing_data.values():
            for episode_id in file_path.keys():
                max_episode = max(max_episode, int(episode_id.split('_')[1]))
        start_episode = max_episode + 1
    else:
        start_episode = 0
    
    # 设置处理范围
    end_episode = len(ds)
    episode_ids = range(start_episode, end_episode)
    
    try:
        # 生成primitive moves数据集
        generate_primitive_dataset(
            builder=dataset_builder,
            episode_ids=episode_ids,
            save_path=output_path
        )
    except Exception as e:
        logging.error(f"生成primitive moves时发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()
