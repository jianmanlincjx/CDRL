CUDA_VISIBLE_DEVICES=0 python renderer/train.py \
    --celeb /data2/JM/code/NED-main/RAVDESS/Train/Actor_01 \
    --checkpoints_dir /data2/JM/code/NED-main/renderer_checkpoints_RAVDESS/Actor_01 \
    --load_pretrain /data2/JM/code/NED-main/renderer_checkpoints/Meta-renderer/checkpoints_meta-renderer \
    --which_epoch 15 \
    --niter 5

CUDA_VISIBLE_DEVICES=0 python renderer/train.py \
    --celeb /data2/JM/code/NED-main/RAVDESS/Train/Actor_02 \
    --checkpoints_dir /data2/JM/code/NED-main/renderer_checkpoints_RAVDESS/Actor_02 \
    --load_pretrain /data2/JM/code/NED-main/renderer_checkpoints/Meta-renderer/checkpoints_meta-renderer \
    --which_epoch 15 \
    --niter 5

CUDA_VISIBLE_DEVICES=0 python renderer/train.py \
    --celeb /data2/JM/code/NED-main/RAVDESS/Train/Actor_03 \
    --checkpoints_dir /data2/JM/code/NED-main/renderer_checkpoints_RAVDESS/Actor_03 \
    --load_pretrain /data2/JM/code/NED-main/renderer_checkpoints/Meta-renderer/checkpoints_meta-renderer \
    --which_epoch 15 \
    --niter 5 

CUDA_VISIBLE_DEVICES=0 python renderer/train.py \
    --celeb /data2/JM/code/NED-main/RAVDESS/Train/Actor_04 \
    --checkpoints_dir /data2/JM/code/NED-main/renderer_checkpoints_RAVDESS/Actor_04 \
    --load_pretrain /data2/JM/code/NED-main/renderer_checkpoints/Meta-renderer/checkpoints_meta-renderer \
    --which_epoch 15 \
    --niter 5 

CUDA_VISIBLE_DEVICES=0 python renderer/train.py \
    --celeb /data2/JM/code/NED-main/RAVDESS/Train/Actor_05 \
    --checkpoints_dir /data2/JM/code/NED-main/renderer_checkpoints_RAVDESS/Actor_05 \
    --load_pretrain /data2/JM/code/NED-main/renderer_checkpoints/Meta-renderer/checkpoints_meta-renderer \
    --which_epoch 15 \
    --niter 5

CUDA_VISIBLE_DEVICES=0 python renderer/train.py \
    --celeb /data2/JM/code/NED-main/RAVDESS/Train/Actor_06 \
    --checkpoints_dir /data2/JM/code/NED-main/renderer_checkpoints_RAVDESS/Actor_06 \
    --load_pretrain /data2/JM/code/NED-main/renderer_checkpoints/Meta-renderer/checkpoints_meta-renderer \
    --which_epoch 15 \
    --niter 5 

# # belindascarberry@task.kim，gUxeeMsWH9BH6