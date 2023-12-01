
accelerate launch training.py \
    --output_dir /fsx/william/styledrop-stage-2 \
    --mixed_precision fp16 \
    --report_to wandb \
    --use_lora \
    --pretrained_model_name_or_path openMUSE/diffusers-pipeline \
    --train_batch_size 8 \
    --lr_scheduler constant \
    --learning_rate 0.00003 \
    --allow_tf32 \
    --validation_prompts \
        'A chihuahua walking on the street in [V] style' \
    --instance_data_dir /fsx/william/amused/good \
    --max_train_steps 1000 \
    --checkpoints_total_limit 20 \
    --checkpointing_steps 500 \
    --validation_steps 100
