#!/bin/bash

python dreambooth/train_dreambooth.py \
    --pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5' \
    --instance_data_dir='data_mixed/data_mixed/map1_fast_sim' \
    --output_dir='output/map1_fast_sim' \
    --instance_prompt='' \
    --resolution=512 \
    --train_batch_size=1 \
    --sample_batch_size=1 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing \
    --learning_rate=2e-06 \
    --lr_scheduler=constant \
    --lr_warmup_steps=1000 \
    --max_train_steps=30000 \
    --mixed_precision=fp16 \
    --report_to=wandb \
    --validation_prompt='a view from a car driving on the road' \
    --validation_steps=200 \
    --checkpointing_steps=1000 \
    --resume_from_checkpoint latest
