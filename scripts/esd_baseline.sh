python ./esd_baseline.py \
    --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
    --removing_concepts \
        "nudity" \
    --validation_prompts \
        "nudity" \
    --num_images_per_prompt 10 \
    --train_batch_size 10 \
    --devices 0 0 \
    --num_train_steps 1000 \
    --finetuning_method noxattn