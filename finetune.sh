export PYTHONPATH="."
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --vla_path "Embodied-CoT/ecot-openvla-7b-bridge" \
  --data_root_dir /home/admin/workspace/data \
  --dataset_name libero_spatial_reasoning \
  --run_root_dir /workspace/checkpoint \
  --adapter_tmp_dir /workspace/checkpoint/adapter_tmp_cross_lora_finetune \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project ECOT_LIBERO \
  --wandb_entity guojy001 \
  --save_steps 200