python ./meta.py \
    --pretrained_model_name_or_path "your-model-path" \
    --removing_concepts \
        "nudity" \
    --num_images_per_prompt 10 \
    --train_batch_size 10 \
    --devices 0 0 \
    --num_train_steps 100 \
    --finetuning_method full \
    --learning_rate 1e-05 \
    --gamma1_1 0.1 \
    --gamma1_2 0.1 \
    --gamma2_1 0.1 \
    --gamma2_1 0.1 \
    --gamma2_1 0.01 \
    --fix_timesteps True \
    --fixed_time_steps 1 2 5 10