celeb=$1
exp_name=$2
checkpoints_dir=$3

python renderer/create_inputs.py --celeb $celeb --exp_name $exp_name --save_shapes
python renderer/test.py --celeb $celeb --exp_name $exp_name --checkpoints_dir $checkpoints_dir --which_epoch 10
# python postprocessing/unalign.py --celeb $celeb --exp_name $exp_name
# python postprocessing/blend.py --celeb $celeb --exp_name $exp_name --save_images


### cross-id
# neutral   FID: 2.354 LSE-D: 9.236 CSIM: 0.821 
# angry     FID: 5.143 LSE-D: 9.399 CSIM: 0.719 
# disgusted FID: 4.812 LSE-D: 9.462 CSIM: 0.810 
# fear      FID: 5.549 LSE-D: 9.207 CSIM: 0.746 
# happy     FID: 4.018 LSE-D: 9.380 CSIM: 0.834 
# sad       FID: 5.534 LSE-D: 9.337 CSIM: 0.736 
# surprised FID: 5.064 LSE-D: 9.227 CSIM: 0.769 