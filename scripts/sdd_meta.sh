python ./sdd_meta.py \
    --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
    --removing_concepts \
        "nudity" \
    --validation_prompts \
        "japan body" \
    --num_images_per_prompt 10 \
    --train_batch_size 10 \
    --devices 0 0 \
    --num_train_steps 1500 \
    --finetuning_method full \
    --gamma1 0.1 \
    --gamma2 0.001 #\
    # --fix_timesteps True