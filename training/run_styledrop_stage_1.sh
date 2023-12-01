
accelerate launch training.py \
    --output_dir /fsx/william/styledrop \
    --mixed_precision fp16 \
    --report_to wandb \
    --use_lora \
    --pretrained_model_name_or_path openMUSE/diffusers-pipeline \
    --train_batch_size 1 \
    --lr_scheduler constant \
    --learning_rate 0.00003 \
    --allow_tf32 \
    --validation_prompts \
        'A chihuahua walking on the street in [V] style' \
    --instance_data_image '/fsx/william/amused/A woman working on a laptop in [V] style.jpg' \
    --max_train_steps 1000 \
    --checkpoints_total_limit 20 \
    --checkpointing_steps 500 \
    --validation_steps 100
