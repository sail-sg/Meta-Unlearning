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
    --gamma1_1 0.1 \
    --gamma1_2 0.01 \
    --gamma2_1 0.1 \
    --gamma2_1 0.1 \
    --gamma2_1 0.01 \
    --fix_timesteps True \
    --fixed_time_steps 1 2 5 10