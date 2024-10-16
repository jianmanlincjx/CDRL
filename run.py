import os
if __name__ == "__main__":
        
    # cross_id
    root = "/data2/JM/code/NED-main/render_test"
    ref_dir = "/data2/JM/code/NED-main/render_reference"
    emotion_list = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    vid_list = sorted(os.listdir(root))
    render_root = "/data2/JM/code/NED-main/renderer_checkpoints"
    drive = {"M003": "W015", "M009": "M003", "W029": "M009", "M012": "W029", "M030": "M012", "W015": "M030"}
    for vid in vid_list:
        celeb_path = os.path.join(root, vid, "neutral")
        for em in emotion_list:
            checkpoint_dir = f"{render_root}/{vid}"
            ref_dirs = os.path.join(ref_dir, drive[vid], em, "DECA")
            os.system(f"python manipulator/test.py \
                            --celeb {celeb_path} \
                            --checkpoints_dir /data2/JM/code/NED-main/manipulator_checkpoints/manipulator_checkpoint_author \
                            --ref_dirs {ref_dirs} \
                            --exp_name {em}_cross_id_dcl_author_manipilator \
                            --which_epoch 5")
            os.system(f"sh ./postprocess.sh \
                            {celeb_path} {em}_cross_id_dcl_author_manipilator {checkpoint_dir}")
            # os.system(f"python postprocessing/images2video.py \
            #                    --imgs_path {celeb_path}/{em}_cross_id_dcl_ce/full_frames \
            #                    --out_path /data2/JM/code/NED-main/result/cross-id/{vid}_{em}_decoulp_learning-manipulator.mp4 ")
            
    ## inter_id
    root = "/data2/JM/code/NED-main/render_test"
    ref_dir = "/data2/JM/code/NED-main/render_reference"
    emotion_list = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    vid_list = sorted(os.listdir(root))

    render_root = "/data2/JM/code/NED-main/renderer_checkpoints"
    drive = {"M003": "M003", "M009": "M009", "W029": "W029", "M012": "M012", "M030": "M030", "W015": "W015"}
    
    for vid in vid_list:
        for em in emotion_list:
            celeb_path = os.path.join(root, vid, em)
            checkpoint_dir = f"{render_root}/{vid}"
            
            ref_dirs = os.path.join(ref_dir, drive[vid], em, "DECA")
            os.system(f"python manipulator/test.py \
                            --celeb {celeb_path} \
                            --checkpoints_dir /data2/JM/code/NED-main/manipulator_checkpoints/manipulator_checkpoint_author \
                            --ref_dirs {ref_dirs} \
                            --exp_name {em}_inter_id_dcl_author_manipilator \
                            --which_epoch 1")
            os.system(f"sh ./postprocess.sh \
                            {celeb_path} {em}_inter_id_dcl_author_manipilator {checkpoint_dir}")
            # os.system(f"python postprocessing/images2video.py \
            #                    --imgs_path {celeb_path}/{em}_cross_id_dcl_ce/full_frames \
            #                    --out_path /data2/JM/code/NED-main/result/inter-id/{vid}_{em}_decoulp_learning-manipulator.mp4 ")
            
