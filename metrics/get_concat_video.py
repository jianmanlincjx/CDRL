import os
import cv2
import copy
from moviepy.editor import *
import random

if __name__ == "__main__":
    pass
    
    ### concat video
    # temp = "inter_id_dcl_1"
    # root = f"/data2/JM/code/NED-main/result_concat/{temp}"
    # save_root = "result_video/inter_id"
    # vid_list = sorted(os.listdir(root))
    # emotion_list = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    # for vid in vid_list:
    #     for em in emotion_list:
    #         vid_em_path = os.path.join(root, vid, em)
    #         video_list = [i for i in sorted(os.listdir(vid_em_path)) if i.endswith(".mp4")]
    #         for video in video_list:
    #             video_path = os.path.join(vid_em_path, video)
    #             source_cross_id = video_path.replace("inter_id_dcl_1", "source_inter_id")
    #             reference_cross_id = video_path.replace("inter_id_dcl_1", "reference_inter_id")
    #             temp = os.path.dirname(reference_cross_id)
    #             ll = random.choice([i for i in os.listdir(temp) if i.endswith(".mp4")])
    #             reference_cross_id = os.path.join(temp, ll)
        
    #             cross_id_scmcl_2 = video_path.replace("inter_id_dcl_1", "inter_id_dcl_1")
    #             cross_id_scmcl_2_author = video_path.replace("inter_id_dcl_1", "inter_id_author")
    #             video_video = clips_array([[VideoFileClip(source_cross_id), VideoFileClip(reference_cross_id), VideoFileClip(cross_id_scmcl_2_author), VideoFileClip(cross_id_scmcl_2)]]) # 左右拼接
    #             video_video.write_videofile(f"{save_root}/{vid}_source_{em}_{video}")
    #             # print(f"{save_root}/{vid}_source_{em}_{video.split(".")[0]}.mp4")
            
    
    temp = "cross_id_dcl_1"
    root = f"/data2/JM/code/NED-main/result_concat/{temp}"
    save_root = "/data2/JM/code/NED-main/result_video/cross-id-author"
    vid_list = sorted(os.listdir(root))
    emotion_list = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    for vid in vid_list:
        for em in emotion_list:
            vid_em_path = os.path.join(root, vid, em)
            video_list = [i for i in sorted(os.listdir(vid_em_path)) if i.endswith(".mp4")]
            for video in video_list:
                video_path = os.path.join(vid_em_path, video)
                source_cross_id = video_path.replace("cross_id_dcl_1", "source_cross_id")
                reference_cross_id = video_path.replace("cross_id_dcl_1", "reference_cross_id")
                temp = os.path.dirname(reference_cross_id)
                ll = random.choice([i for i in os.listdir(temp) if i.endswith(".mp4")])
                reference_cross_id = os.path.join(temp, ll)
        
                cross_id_scmcl_2 = video_path.replace("cross_id_dcl_1", "cross_id_dcl_1")
                cross_id_scmcl_2_author = video_path.replace("cross_id_dcl_1", "cross_id_author")
                video_video = clips_array([[VideoFileClip(source_cross_id), VideoFileClip(reference_cross_id), VideoFileClip(cross_id_scmcl_2_author)]]) # 左右拼接
                video_video.write_videofile(f"{save_root}/{vid}_source_{em}_{video}")
                # print(f"{save_root}/{vid}_source_{em}_{video.split(".")[0]}.mp4")

    ## image2video
    # shuffx = "inter_id_dcl_1"
    # root = f"/data2/JM/code/NED-main/result_concat/{shuffx}"
    # vid_list = sorted(os.listdir(root))
    # emotion_list = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    # for vid in vid_list:
    #     for em in emotion_list:
    #         vid_em_path = os.path.join(root, vid, em)
    #         video_list = [i for i in sorted(os.listdir(vid_em_path)) if not i.endswith(".mp4")]
    
    #         for video in video_list:
    #             video_path = os.path.join(vid_em_path, video)
    #             save_path = copy.deepcopy(video_path)+".mp4"
    #             os.system(f"ffmpeg -y -r 30 -pattern_type glob -i '{video_path}/*.png' -c:v libx264 -b:v 10M {save_path}")
    
    # reference video inter-id
    # shuffx = "reference_inter_id"
    # save_root = f"/data2/JM/code/NED-main/result_concat/{shuffx}"
    # root = "/data2/JM/code/NED-main/render_test"
    # vid_list = sorted(os.listdir(root))
    # drive = {"M003": "W015", "M009": "M003", "W029": "M009", "M012": "W029", "M030": "M012", "W015": "M030"}
    # emotion_list = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    # for vid in vid_list:
    #     for em in emotion_list: 
    #         vid_path = os.path.join(root, vid, em)
    #         video_list = [i for i in sorted(os.listdir(os.path.join(vid_path, "videos"))) if i.endswith(".mp4")]
    #         img_path = os.path.join(vid_path, "images")
    #         img_list = sorted(os.listdir(img_path))
    #         video_frame_all = 0
    #         for video in video_list:
    #             video_path = os.path.join(vid_path, "videos", video)
    #             video_name = video.split(".")[0]
    #             cap = cv2.VideoCapture(video_path)
    #             frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #             start_idx = video_frame_all
    #             end_idx = start_idx + frame_count
    #             for idx in range(start_idx, end_idx):
    #                 img_idx = img_list[idx]
    #                 cp_path = os.path.join(img_path, img_idx)
    #                 save_path = os.path.join(save_root, vid, em, video_name)
    #                 os.makedirs(save_path, exist_ok=True)
    #                 os.system(f"cp {cp_path} {save_path}")
    #             video_frame_all += frame_count   
    
    
    
    # reference video cross-id
    # shuffx = "reference_cross_id"
    # save_root = f"/data2/JM/code/NED-main/result_concat/{shuffx}"
    # root = "/data2/JM/code/NED-main/render_test"
    # vid_list = sorted(os.listdir(root))
    # drive = {"M003": "W015", "M009": "M003", "W029": "M009", "M012": "W029", "M030": "M012", "W015": "M030"}
    # emotion_list = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    # for vid in vid_list:
    #     for em in emotion_list: 
    #         vid_path = os.path.join(root, drive[vid], em)
    #         video_list = [i for i in sorted(os.listdir(os.path.join(vid_path, "videos"))) if i.endswith(".mp4")]
    #         img_path = os.path.join(vid_path, "images")
    #         img_list = sorted(os.listdir(img_path))
    #         video_frame_all = 0
    #         for video in video_list:
    #             video_path = os.path.join(vid_path, "videos", video)
    #             video_name = video.split(".")[0]
    #             cap = cv2.VideoCapture(video_path)
    #             frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #             start_idx = video_frame_all
    #             end_idx = start_idx + frame_count
    #             for idx in range(start_idx, end_idx):
    #                 img_idx = img_list[idx]
    #                 cp_path = os.path.join(img_path, img_idx)
    #                 save_path = os.path.join(save_root, vid, em, video_name)
    #                 os.makedirs(save_path, exist_ok=True)
    #                 os.system(f"cp {cp_path} {save_path}")
    #             video_frame_all += frame_count
    
    # # source video corss-id
    # shuffx = "source_cross_id"
    # save_root = f"/data2/JM/code/NED-main/result_concat/{shuffx}"
    # root = "/data2/JM/code/NED-main/render_test"
    # vid_list = sorted(os.listdir(root))
    # em_cross = "neutral"
    # emotion_list = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    # for vid in vid_list:
    #     vid_path = os.path.join(root, vid, em_cross)
    #     for em in emotion_list: 
    #         video_list = [i for i in sorted(os.listdir(os.path.join(vid_path, "videos"))) if i.endswith(".mp4")]
    #         img_path = os.path.join(vid_path, "images")
    #         img_list = sorted(os.listdir(img_path))
    #         video_frame_all = 0
    #         for video in video_list:
    #             video_path = os.path.join(vid_path, "videos", video)
    #             video_name = video.split(".")[0]
    #             cap = cv2.VideoCapture(video_path)
    #             frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #             start_idx = video_frame_all
    #             end_idx = start_idx + frame_count
    #             for idx in range(start_idx, end_idx):
    #                 img_idx = img_list[idx]
    #                 cp_path = os.path.join(img_path, img_idx)
    #                 save_path = os.path.join(save_root, vid, em, video_name)
    #                 os.makedirs(save_path, exist_ok=True)
    #                 os.system(f"cp {cp_path} {save_path}")
    #             video_frame_all += frame_count
    
    
    # # source video inter-id
    # shuffx = "source_inter_id"
    # save_root = f"/data2/JM/code/NED-main/result_concat/{shuffx}"
    # root = "/data2/JM/code/NED-main/render_test"
    # vid_list = sorted(os.listdir(root))
    # emotion_list = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    # for vid in vid_list:
    #     for em in emotion_list: 
    #         vid_path = os.path.join(root, vid, em)
    #         video_list = [i for i in sorted(os.listdir(os.path.join(vid_path, "videos"))) if i.endswith(".mp4")]
    #         img_path = os.path.join(vid_path, "images")
    #         img_list = sorted(os.listdir(img_path))
    #         video_frame_all = 0
    #         for video in video_list:
    #             video_path = os.path.join(vid_path, "videos", video)
    #             video_name = video.split(".")[0]
    #             cap = cv2.VideoCapture(video_path)
    #             frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #             start_idx = video_frame_all
    #             end_idx = start_idx + frame_count
    #             for idx in range(start_idx, end_idx):
    #                 img_idx = img_list[idx]
    #                 cp_path = os.path.join(img_path, img_idx)
    #                 save_path = os.path.join(save_root, vid, em, video_name)
    #                 os.makedirs(save_path, exist_ok=True)
    #                 os.system(f"cp {cp_path} {save_path}")
    #             video_frame_all += frame_count
    
    
    
    
    # ####cross_id
    ####需要修改的##
    # shuffx = "cross_id_dcl_1"
    # save_root = f"/data2/JM/code/NED-main/result_concat/{shuffx}"
    # ##############
    # ## cross_id 
    # root = "/data2/JM/code/NED-main/render_test"
    # vid_list = sorted(os.listdir(root))
    # em_cross = "neutral"
    # emotion_list = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    # for vid in vid_list:
    #     vid_path = os.path.join(root, vid, em_cross)
    #     for em in emotion_list:
    #         video_list = [i for i in sorted(os.listdir(os.path.join(vid_path, "videos"))) if i.endswith(".mp4")]
    #         img_path = os.path.join(vid_path, f"{em}_{shuffx}", "images")
    #         img_list = sorted(os.listdir(img_path))
    #         video_frame_all = 0
    #         for video in video_list:
    #             video_path = os.path.join(vid_path, "videos", video)
    #             video_name = video.split(".")[0]
    #             cap = cv2.VideoCapture(video_path)
    #             frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #             start_idx = video_frame_all
    #             end_idx = start_idx + frame_count
    #             for idx in range(start_idx, end_idx-2):
    #                 img_idx = img_list[idx]
    #                 cp_path = os.path.join(img_path, img_idx)
    #                 save_path = os.path.join(save_root, vid, em, video_name)
    #                 os.makedirs(save_path, exist_ok=True)
    #                 os.system(f"cp {cp_path} {save_path}")
    #             video_frame_all += frame_count
                                         

    # ####inter_id
    # ####需要修改的##
    # shuffx = "inter_id_dcl_1"
    # save_root = f"/data2/JM/code/NED-main/result_concat/{shuffx}"
    # ##############
    # ## cross_id 
    # root = "/data2/JM/code/NED-main/render_test"
    # vid_list = sorted(os.listdir(root))
    # emotion_list = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    # for vid in vid_list:
    #     for em in emotion_list:
    #         vid_path_em = os.path.join(root, vid, em)
    #         vid_path = os.path.join(root, vid, em, f"{em}_{shuffx}")
    #         video_list = [i for i in sorted(os.listdir(os.path.join(vid_path_em, "videos"))) if i.endswith(".mp4")]
    #         img_path = os.path.join(vid_path, "images")
    #         img_list = sorted(os.listdir(img_path))
    #         video_frame_all = 0
    #         for video in video_list:
    #             video_path = os.path.join(vid_path_em, "videos", video)
    #             video_name = video.split(".")[0]
    #             cap = cv2.VideoCapture(video_path)
    #             frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #             start_idx = video_frame_all
    #             end_idx = start_idx + frame_count
    #             for idx in range(start_idx, end_idx-2):
    #                 img_idx = img_list[idx]
    #                 cp_path = os.path.join(img_path, img_idx)
    #                 save_path = os.path.join(save_root, vid, em, video_name)
    #                 os.makedirs(save_path, exist_ok=True)
    #                 os.system(f"cp {cp_path} {save_path}")
    #             video_frame_all += frame_count
            
            
    