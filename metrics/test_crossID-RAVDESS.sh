manipulator_ckpt=${1:-manipulator_checkpoint}
renderer_ckpt=${2:-renderer_checkpoints_RAVDESS}
renderer_epoch=${3:-10}
tag=${4:-scmcl-RAVDESS-6ID}

celeb='/data2/JM/code/NED-main/RAVDESS'

for emo in neutral angry disgusted fear happy sad surprised
do
    rm -rf ${celeb}/test/neutral/Actor_01/ref_on_${emo}_Actor_06_${tag}
    python manipulator/test.py --celeb ${celeb}/test/neutral/Actor_01/ --checkpoints_dir manipulator_checkpoints/${manipulator_ckpt} --which_epoch 1 --ref_dirs ${celeb}/ref/${emo}/Actor_06/DECA --exp_name ref_on_${emo}_Actor_06_${tag}
    sh postprocess.sh ${celeb}/test/neutral/Actor_01/ ref_on_${emo}_Actor_06_${tag} /data2/JM/code/NED-main/${renderer_ckpt}/Actor_01 ${renderer_epoch}

    rm -rf ${celeb}/test/neutral/Actor_02/ref_on_${emo}_Actor_01_${tag}
    python manipulator/test.py --celeb ${celeb}/test/neutral/Actor_02/ --checkpoints_dir manipulator_checkpoints/${manipulator_ckpt} --which_epoch 1 --ref_dirs ${celeb}/ref/${emo}/Actor_01/DECA --exp_name ref_on_${emo}_Actor_01_${tag}
    sh postprocess.sh ${celeb}/test/neutral/Actor_02/ ref_on_${emo}_Actor_01_${tag} /data2/JM/code/NED-main/${renderer_ckpt}/Actor_02 ${renderer_epoch}

    rm -rf ${celeb}/test/neutral/Actor_03/ref_on_${emo}_Actor_02_${tag}
    python manipulator/test.py --celeb ${celeb}/test/neutral/Actor_03/ --checkpoints_dir manipulator_checkpoints/${manipulator_ckpt} --which_epoch 1 --ref_dirs ${celeb}/ref/${emo}/Actor_02/DECA --exp_name ref_on_${emo}_Actor_02_${tag}
    sh postprocess.sh ${celeb}/test/neutral/Actor_03/ ref_on_${emo}_Actor_02_${tag} /data2/JM/code/NED-main/${renderer_ckpt}/Actor_03 ${renderer_epoch}

    rm -rf ${celeb}/test/neutral/Actor_04/ref_on_${emo}_Actor_03_${tag}
    python manipulator/test.py --celeb ${celeb}/test/neutral/Actor_04/ --checkpoints_dir manipulator_checkpoints/${manipulator_ckpt} --which_epoch 1 --ref_dirs ${celeb}/ref/${emo}/Actor_03/DECA --exp_name ref_on_${emo}_Actor_03_${tag}
    sh postprocess.sh ${celeb}/test/neutral/Actor_04/ ref_on_${emo}_Actor_03_${tag} /data2/JM/code/NED-main/${renderer_ckpt}/Actor_04 ${renderer_epoch}

    rm -rf ${celeb}/test/neutral/Actor_05/ref_on_${emo}_Actor_04_${tag}
    python manipulator/test.py --celeb ${celeb}/test/neutral/Actor_05/ --checkpoints_dir manipulator_checkpoints/${manipulator_ckpt} --which_epoch 1 --ref_dirs ${celeb}/ref/${emo}/Actor_04/DECA --exp_name ref_on_${emo}_Actor_04_${tag}
    sh postprocess.sh ${celeb}/test/neutral/Actor_05/ ref_on_${emo}_Actor_04_${tag} /data2/JM/code/NED-main/${renderer_ckpt}/Actor_05 ${renderer_epoch}

    rm -rf ${celeb}/test/neutral/Actor_06/ref_on_${emo}_Actor_05_${tag}
    python manipulator/test.py --celeb ${celeb}/test/neutral/Actor_06/ --checkpoints_dir manipulator_checkpoints/${manipulator_ckpt} --which_epoch 1 --ref_dirs ${celeb}/ref/${emo}/Actor_05/DECA --exp_name ref_on_${emo}_Actor_05_${tag}
    sh postprocess.sh ${celeb}/test/neutral/Actor_06/ ref_on_${emo}_Actor_05_${tag} /data2/JM/code/NED-main/${renderer_ckpt}/Actor_06 ${renderer_epoch}
done
