import os
from tqdm import tqdm


if __name__ == "__main__":
    
    vid_list = ["M003", "M009", "M012", "M030", "W015", "W029"]
    emotion_list = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    root = "/data2/JM/code/NED-main/result/manipulator_5"
    ss = "manipulator_5"
    sub_list = ["corss-id", "inter-id"]
    
    for sub in sub_list:
        for vid in vid_list:
            for emotion in tqdm(emotion_list):
                save_path = f"/data2/JM/code/NED-main/result/images/{ss}/{sub}/{vid}/{emotion}"
                os.makedirs(save_path, exist_ok=True)
                video_path = f"{root}/cross-id/{vid}_{emotion}_decoulp_learning-manipulator.mp4"
                os.system(f"ffmpeg -i {video_path} {save_path}/%06d.png")
                
    vid_list = ["M003", "M009", "M012", "M030", "W015", "W029"]
    emotion_list = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    root = "/data2/JM/code/NED-main/result/manipulator_3"
    ss = os.path.basename(root)
    sub_list = ["corss-id", "inter-id"]
    
    for sub in sub_list:
        for vid in vid_list:
            for emotion in tqdm(emotion_list):
                save_path = f"/data2/JM/code/NED-main/result/images/{ss}/{sub}/{vid}/{emotion}"
                os.makedirs(save_path, exist_ok=True)
                video_path = f"{root}/cross-id/{vid}_{emotion}_decoulp_learning-manipulator.mp4"
                os.system(f"ffmpeg -i {video_path} {save_path}/%06d.png")
                
    vid_list = ["M003", "M009", "M012", "M030", "W015", "W029"]
    emotion_list = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    root = "/data2/JM/code/NED-main/result/manipulator_2"
    ss = os.path.basename(root)
    sub_list = ["corss-id", "inter-id"]
    
    for sub in sub_list:
        for vid in vid_list:
            for emotion in tqdm(emotion_list):
                save_path = f"/data2/JM/code/NED-main/result/images/{ss}/{sub}/{vid}/{emotion}"
                os.makedirs(save_path, exist_ok=True)
                video_path = f"{root}/cross-id/{vid}_{emotion}_decoulp_learning-manipulator.mp4"
                os.system(f"ffmpeg -i {video_path} {save_path}/%06d.png")
                
    vid_list = ["M003", "M009", "M012", "M030", "W015", "W029"]
    emotion_list = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    root = "/data2/JM/code/NED-main/result/manipulator_1"
    ss = os.path.basename(root)
    sub_list = ["corss-id", "inter-id"]
    for sub in sub_list:
        for vid in vid_list:
            for emotion in tqdm(emotion_list):
                save_path = f"/data2/JM/code/NED-main/result/images/{ss}/{sub}/{vid}/{emotion}"
                os.makedirs(save_path, exist_ok=True)
                video_path = f"{root}/cross-id/{vid}_{emotion}_decoulp_learning-manipulator.mp4"
                os.system(f"ffmpeg -i {video_path} {save_path}/%06d.png")
                
    vid_list = ["M003", "M009", "M012", "M030", "W015", "W029"]
    emotion_list = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    root = "/data2/JM/code/NED-main/result/manipulator_1_5"
    ss = os.path.basename(root)
    sub_list = ["corss-id", "inter-id"]
    for sub in sub_list:
        for vid in vid_list:
            for emotion in tqdm(emotion_list):
                save_path = f"/data2/JM/code/NED-main/result/images/{ss}/{sub}/{vid}/{emotion}"
                os.makedirs(save_path, exist_ok=True)
                video_path = f"{root}/cross-id/{vid}_{emotion}_decoulp_learning-manipulator.mp4"
                os.system(f"ffmpeg -i {video_path} {save_path}/%06d.png")