from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

def merge_lora_to_base_model(
    base_model_path: str,
    lora_path: str,
    output_path: str
):
    """
    合并基础模型和LoRA权重
    
    Args:
        base_model_path: 基础模型路径
        lora_path: LoRA adapter路径 
        output_path: 合并后的输出模型路径
    """
    # 加载基础模型
    print(f"正在加载基础模型: {base_model_path}")
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    # 加载LoRA模型
    print(f"正在加载LoRA模型: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    # 合并LoRA权重到基础模型
    print("正在合并模型...")
    model = model.merge_and_unload()
    
    # 保存合并后的模型
    print(f"正在保存合并后的模型到: {output_path}")
    model.save_pretrained(output_path)
    
    # 同时保存tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    
    print("完成!")

if __name__ == "__main__":
    # 使用示例
    merge_lora_to_base_model(
        base_model_path="Embodied-CoT/ecot-openvla-7b-bridge",  # 替换为你的基础模型路径
        lora_path="/workspace/checkpoint/adapter_tmp_cross_lora_finetune/ecot-openvla-7b-bridge+libero_spatial_reasoning+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug",  # 替换为你的LoRA adapter路径
        output_path="/workspace/checkpoint/ecot-openvla-7b-libero-spatial"  # 替换为你想要保存的路径
    )