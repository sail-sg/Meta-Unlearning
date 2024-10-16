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
    --gamma1 0.1 \
    --gamma2 0.001