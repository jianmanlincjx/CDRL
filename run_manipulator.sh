CUDA_VISIBLE_DEVICES=3 python manipulator/train.py \
    --train_root MEAD_data/ \
    --selected_actors M003 M009 M012 M030 W015 W029 \
    --selected_actors_val W029 \
    --checkpoints_dir /data2/JM/code/NED-main/manipulator_checkpoints/manipulator_checkpoint \
    --finetune