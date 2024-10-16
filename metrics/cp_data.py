import os
import cv2
from tqdm import tqdm
from moviepy.editor import VideoFileClip, clips_array

if __name__ == "__main__":
    
    ## Concat video Source Reference ASCCL NED
    # Source_root = "/data2/JM/code/NED-main/RAVDESS_data/Inter-id"
    # Target_root = "/data2/JM/code/NED-main/RAVDESS_data/Inter-id_video"
    # ASCCL_path = "/data2/JM/code/NED-main/RAVDESS_data/Inter-id/ASCCL_plus"
    # video_list = sorted(os.listdir(ASCCL_path))
    # video_list_new = sorted(os.listdir("/data2/JM/code/NED-main/RAVDESS_data/Inter-id/Source_video"))
    # video_list_new_ = sorted(os.listdir("/data2/JM/code/NED-main/RAVDESS_data/Inter-id/Baseline"))
    # for idx, videoname in enumerate(video_list):
    #     ASCCL_video = os.path.join(ASCCL_path, videoname)
    #     # Source_video = ASCCL_video.replace("ASCCL_plus", "Source_video").replace("_plus", "")
    #     # Driven_video = ASCCL_video.replace("ASCCL_plus", "Driven_video").replace("_plus", "")
    #     Baseline_video = f"/data2/JM/code/NED-main/RAVDESS_data/Inter-id/Baseline/{video_list_new_[idx]}"
    #     Source_video = f"/data2/JM/code/NED-main/RAVDESS_data/Inter-id/Source_video/{video_list_new[idx]}"
    #     Driven_video = f"/data2/JM/code/NED-main/RAVDESS_data/Inter-id/Driven_video/{video_list_new[idx]}"

    #     video = clips_array([[VideoFileClip(Source_video), VideoFileClip(Driven_video), VideoFileClip(ASCCL_video), VideoFileClip(Baseline_video)]]) # 左右拼接

    #     video.write_videofile(f"{Target_root}/{videoname}", codec="libx264", bitrate="10M")
    ## RIFE
    # root = "/data2/JM/code/NED-main/RAVDESS_data/Cross-id/ASCCL_plus"
    # save_root = "/data2/JM/code/NED-main/RAVDESS_data/Cross-id/ASCCL_120"
    # video_list = sorted(os.listdir(root))
    # # _4X_120fps

    # for video in video_list:
    #     video_path = os.path.join(root, video)
    #     os.system(f"CUDA_VISIBLE_DEVICES=2 python3 /data2/JM/code/NED-main/metrics/ECCV2022-RIFE/inference_video.py \
    #                 --exp=2 \
    #                 --video={video_path}")
    #     video_name = video_path.split(".")[0]

    #     os.system(f"ffmpeg -i {video_name}_4X_120fps.mp4 -b:v 10M {save_root}/{video}")
        
    # root = "/data2/JM/code/NED-main/RAVDESS_data/Inter-id/ASCCL"
    # video_list = sorted(os.listdir(root))
    # for video in tqdm(video_list):
    #     video_path = os.path.join(root, video)
    #     new_video_path = video_path.replace("ASCCL", "ASCCL_plus")
    #     vid = os.path.basename(new_video_path)[:8]
    #     em = os.path.basename(new_video_path)[9:-15]

    #     Source_video = f"/data2/JM/code/NED-main/RAVDESS/test/{vid}/audios/{em}_02_02.wav"
    #     os.system(f"ffmpeg -i {video_path} -i {Source_video} -c:v copy -map 0:v:0 -map 1:a:0 -b:v 5M {new_video_path} -y")
    
    
        
    ## CP_cross-id_Source
    # Source_root = "/data2/JM/code/NED-main/RAVDESS/test/neutral"
    # Target_root = "/data2/JM/code/NED-main/RAVDESS_data/Cross-id/Source_video"
    # vid_list = ["Actor_01", "Actor_02", "Actor_03", "Actor_04", "Actor_05", "Actor_06"]
    # emotion_list = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    # for vid in vid_list:
    #     for emotion in tqdm(emotion_list):
    #         Source_image = os.path.join(Source_root, vid, "images")
    #         targte_path = os.path.join(Target_root, f"{vid}_ASCCL_video_{emotion}.mp4")
    #         os.system(f"ffmpeg -r 30 -i {Source_image}/%06d.png -b:v 10M {targte_path}")
    
    ## CP_cross-id_Driven
    # Source_root = "/data2/JM/code/NED-main/RAVDESS/ref"
    # Target_root = "/data2/JM/code/NED-main/RAVDESS_data/Cross-id/Driven_video"
    # vid_list = ["Actor_01", "Actor_02", "Actor_03", "Actor_04", "Actor_05", "Actor_06"]
    # drive = {"Actor_01": "Actor_06", "Actor_02": "Actor_01", "Actor_03": "Actor_02", "Actor_04": "Actor_03", "Actor_05": "Actor_04", "Actor_06": "Actor_05"}
    # emotion_list = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    # for vids in vid_list:
    #     vid = drive[vids]
    #     for emotion in tqdm(emotion_list):
    #         Source_image = os.path.join(Source_root, emotion, vid, "images")
    #         targte_path = os.path.join(Target_root, f"{vids}_ASCCL_video_{emotion}.mp4")
    #         os.system(f"ffmpeg -r 30 -i {Source_image}/%06d.png -b:v 10M {targte_path}")


    ## CP_inter-id_Source
    # Source_root = "/data2/JM/code/NED-main/RAVDESS/test/inter-id"
    # Target_root = "/data2/JM/code/NED-main/RAVDESS_data/Inter-id/Source_video"
    # vid_list = ["Actor_01", "Actor_02", "Actor_03", "Actor_04", "Actor_05", "Actor_06"]
    # emotion_list = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    # for vid in vid_list:
    #     for emotion in tqdm(emotion_list):
    #         Source_image = os.path.join(Source_root, vid, emotion, "images")
    #         targte_path = os.path.join(Target_root, f"{vid}_ASCCL_video_{emotion}.mp4")
    #         os.system(f"ffmpeg -r 30 -i {Source_image}/%06d.png -b:v 10M {targte_path}")
    
    ## CP_inter-id_Driven
    # Source_root = "/data2/JM/code/NED-main/RAVDESS/test/inter-id"
    # Target_root = "/data2/JM/code/NED-main/RAVDESS_data/Inter-id/Driven_video"
    # vid_list = ["Actor_01", "Actor_02", "Actor_03", "Actor_04", "Actor_05", "Actor_06"]
    # emotion_list = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    # for vid in vid_list:
    #     for emotion in tqdm(emotion_list):
    #         Source_image = os.path.join(Source_root, vid, emotion, "images")
    #         targte_path = os.path.join(Target_root, f"{vid}_ASCCL_video_{emotion}.mp4")
    #         os.system(f"ffmpeg -r 30 -i {Source_image}/%06d.png -b:v 10M {targte_path}")
    
    ## CP 2 inter-id
    root = "/data2/JM/code/NED-main/RAVDESS/test"
    save_root = "/data2/JM/code/NED-main/RAVDESS/test/inter-id"
    vid_list = ["Actor_01", "Actor_02", "Actor_03", "Actor_04", "Actor_05", "Actor_06"]
    emotion_list = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    fld_name = ['eye_landmarks_aligned', 'shapes_aligned', 'nmfcs', 'align_transforms', 'full_frames', 'masks', 'shapes', 'masks_aligned', 'images', 'nmfcs_aligned', 'DECA', 'eye_landmarks', 'faces', 'faces_aligned']
    # fld_name = ['videos']
    for vid in vid_list:
        vid_path = f"{root}/{vid}"
        for fld in fld_name:
            idx = 0
            for emotion in emotion_list:
                source_path = f"{root}/{vid}/{fld}"
                source_fld_list = sorted(os.listdir(source_path))
                save_path = f"{save_root}/{vid}/{emotion}/{fld}"
                os.makedirs(save_path, exist_ok=True)
                # videos = f"{vid_path}/videos/{emotion}_02_02.mp4"
                # txt = f"{vid_path}/videos/{emotion}_02_02.txt"
                # os.system(f"cp {videos} {save_root}/{vid}/{emotion}/videos/2.mp4")
                # os.system(f"cp {txt} {save_root}/{vid}/{emotion}/videos/2.txt")
                video_path = f"{vid_path}/videos/{emotion}_02_02.mp4"
                video_capture = cv2.VideoCapture(video_path)
                frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_count += idx
                for index, i in tqdm(enumerate(range(idx, frame_count-1))):
                    name = source_fld_list[i]
                    name_last = name.split(".")[1]
                    name_first = str(index).zfill(6)+f".{name_last}"
                    os.system(f"cp {source_path}/{name} {save_path}/{name_first}")
                    idx += 1
            
                    
