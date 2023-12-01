
accelerate launch training.py \
    --output_dir /fsx/william/styledrop-lora-dreambooth-2 \
    --mixed_precision fp16 \
    --report_to wandb \
    --resume_from_checkpoint latest \
    --seed 9345104 \
    --is_lora \
    --pretrained_model_name_or_path openMUSE/diffusers-pipeline \
    --train_batch_size 6 \
    --lr_scheduler constant_with_warmup \
    --lr_warmup_steps 2000 \
    --learning_rate 0.0003 \
    --allow_tf32 \
    --validation_prompts \
        'a photo sks getting a haircut' \
        'a photo sks on mount fuji' \
        'a photo sks on a skateboard' \
        'a photo sks on a swing' \
        'a photo sks on a table' \
        'a photo sks sleeping' \
        'a photo of sks backpack' \
        'a photo of sks riding a bike' \
    --instance_data_dir /fsx/william/cat_toy \
    --instance_prompt 'A photo of sks toy' \
    --max_train_steps 2000 \
    --checkpoints_total_limit 20 \
    --checkpointing_steps 100
