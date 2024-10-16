actor=${1:-Actor_05}
reference=${2:-Actor_05}
manipulator_ckpt=${3:-manipulator_checkpoints_pretrained_affwild2_TCCL}
manipulator_epoch=${5:-2}
renderer_epoch=${6:-10}
tag=${7:-avcl-6ID-tmpr-render}

celeb='/data2/JM/code/NED-main/RAVDESS'

rm -rf ${celeb}/test/${actor}/ref_on_${reference}_${tag}

python manipulator/test.py --celeb ${celeb}/test/${actor}/ --checkpoints_dir manipulator_checkpoints/${manipulator_ckpt} --which_epoch ${manipulator_epoch} --ref_dirs ${celeb}/test/${reference}/DECA --exp_name ref_on_${reference}_${tag}

sh postprocess.sh ${celeb}/test/${actor}/ ref_on_${reference}_${tag} RAVDESS_checkpoint/${actor} ${renderer_epoch}

mkdir -p celeb/out_videos/${actor}

python postprocessing/images2video.py --imgs_path celeb/test/${actor}/ref_on_${reference}_${tag}/full_frames --out_path celeb/out_videos/${actor}/${actor}_ref_on_${reference}_${tag}

# for actor in Actor_01 Actor_02 Actor_03 Actor_04 Actor_05 Actor_06; do ./scripts/test_interID-RAVDESS.sh $actor $actor baseline-RAVDESS-6ID $actor 05 5 baseline-6ID; done

# for actor in Actor_01 Actor_02 Actor_03 Actor_04 Actor_05 Actor_06; do ./scripts/test_interID-RAVDESS.sh $actor $actor scmcl-RAVDESS-6ID ${actor}_scmcl/export 05 01 scmcl-RAVDESS-6ID; done
