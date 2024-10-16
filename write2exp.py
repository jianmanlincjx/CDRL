import os
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import cv2

if __name__ == "__main__":
    # # get audio inter id
    # root = "/data2/JM/code/NED-main/render_reference"
    # vid_list = sorted(os.listdir(root))
    # emotion_list = [ "neutral", "angry", "disgusted", "fear", "happy", "sad", "surprised"]
    
    # for vid in vid_list:
    #     for em in emotion_list:
    #         video_path = os.path.join(root, vid, em, "videos")
    #         video_list = [i for i in sorted(os.listdir(video_path)) if i.endswith(".mp4")]
    #         for video in video_list:
    #             audio_path = os.path.join(video_path, video)
          
    #             audio = VideoFileClip(audio_path)
    #             audio_new = audio.audio
    #             save_path = os.path.join("/data2/JM/code/NED-main/exp/txt", vid, "audios_inter_id", em+"_"+video.split(".")[0]+".m4a")
      
    #             os.makedirs(os.path.join("/data2/JM/code/NED-main/exp/txt", vid, "audios_inter_id"), exist_ok=True)
    #             audio_new.write_audiofile(save_path, codec='aac')
    
    
    
    # # get audio cross id
    # root = "/data2/JM/code/NED-main/render_test"
    # vid_list = sorted(os.listdir(root))
    # emotion_list = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    
    # for vid in vid_list:
    #     video_path = os.path.join(root, vid, "neutral", "videos")
    #     video_list = [i for i in sorted(os.listdir(video_path)) if i.endswith(".mp4")]
    #     for video in video_list:
    #         audio_path = os.path.join(video_path, video)
    #         audio = VideoFileClip(audio_path)
    #         audio_new = audio.audio
    #         for em in emotion_list:
    #             save_path = os.path.join("/data2/JM/code/NED-main/exp/txt", vid, "audios", em+"_"+video.split(".")[0]+".m4a")
    #             os.makedirs(os.path.join("/data2/JM/code/NED-main/exp/txt", vid, "audios"), exist_ok=True)
    #             audio_new.write_audiofile(save_path, codec='aac')
    
    # ## write txt cross_id
    # root = "/data2/JM/code/NED-main/render_test"
    # vid_list = sorted(os.listdir(root))
    # txt_root = "/data2/JM/code/NED-main/exp/txt"
    # emotion_list = ["angry_cross_id_scmcl",  "disgusted_cross_id_scmcl", "fear_cross_id_scmcl", "happy_cross_id_scmcl", "neutral_cross_id_scmcl", "sad_cross_id_scmcl", "surprised_cross_id_scmcl"]
    # for vid in vid_list:
    #     os.makedirs(os.path.join(txt_root, vid, "videos"), exist_ok=True)
    #     idx = 0
    #     for em in emotion_list:
    #         em_vid_path = os.path.join(root, vid, "neutral", em, "images")
    #         img_len_all = len(os.listdir(em_vid_path))

    #         video_path = os.path.join(root, vid, "neutral", "videos")
    #         video_list = [i for i in sorted(os.listdir(video_path)) if i.endswith(".mp4")]
    #         video_list_count = 0
    #         for video in video_list:
    
    #             video_path_ = os.path.join(video_path, video)
    #             cap = cv2.VideoCapture(video_path_)
    #             frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #             video_list_count += frame_count
    #             for i in range(frame_count):
    #                 with open(os.path.join(txt_root, vid, "videos/_frame_info.txt"), "a") as file:
    #                     # em_ = em.split("_")[0]
    #                     # video_name = os.path.basename(video_path_).split(".")[0]
    #                     # file.write(f"{em_}_{video_name}_{i} {idx}"+"\n")
    #                     idx += 1
    #         print(f"{vid} {em} {idx - video_list_count} {idx}")
    
    
    # ## write txt inter_id
    # root = "/data2/JM/code/NED-main/render_test"
    # vid_list = sorted(os.listdir(root))
    # txt_root = "/data2/JM/code/NED-main/exp/txt"
    # emotion_list = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    # for vid in vid_list:
    #     os.makedirs(os.path.join(txt_root, vid, "videos"), exist_ok=True)
    #     idx = 0
    #     for em in emotion_list:
    #         em_vid_path = os.path.join(root, vid, em, f"{em}_cross_id_scmcl_2_inter_ID", "images")
    #         img_len_all = len(os.listdir(em_vid_path))

    #         video_path = os.path.join(root, vid, em, "videos")
    #         video_list = [i for i in sorted(os.listdir(video_path)) if i.endswith(".mp4")]
    #         video_list_count = 0
    #         for video in video_list:
    #             video_path_ = os.path.join(video_path, video)
    #             cap = cv2.VideoCapture(video_path_)
    #             frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #             video_list_count += frame_count
    #             for i in range(frame_count):
    #                 with open(os.path.join(txt_root, vid, "videos/_frame_info_inter_id.txt"), "a") as file:
    #                     # em_ = em.split("_")[0]
    #                     # video_name = os.path.basename(video_path_).split(".")[0]
    #                     # file.write(f"{em_}_{video_name}_{i} {idx}"+"\n")
    #                     idx += 1
                        
    #         print(f"{vid} {em} {idx - video_list_count} {idx}")
    
    
    ### 2 exp/cmp_crossID/baseline/gen-inter-id
    root_gen = "/data2/JM/code/NED-main/exp/cmp_inter/gen"
    root_ori = "/data2/JM/code/NED-main/render_test"
    vid_list = sorted(os.listdir(root_ori))
    emotion_list_ori = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    for vid in vid_list:
        vid_path = os.path.join(root_ori, vid)
        for idx, em in enumerate(emotion_list_ori):
            vid_em_path = os.path.join(vid_path, em, f"{em}_inter_id_dcl_author_manipilator", "faces_aligned")
            new_path = os.path.join(root_gen, emotion_list_ori[idx], vid)

            os.makedirs(new_path, exist_ok=True)
            os.system(f"cp {vid_em_path}/*.png {new_path}")
    
    
    
    # ### 2 exp/cmp_crossID/baseline/gen-cross-id
    root_gen = "/data2/JM/code/NED-main/exp/cmp_crossID/gen"
    root_ori = "/data2/JM/code/NED-main/render_test"
    vid_list = sorted(os.listdir(root_ori))
    emotion_list_ori = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    for vid in vid_list:
        vid_path = os.path.join(root_ori, vid)
        for idx, em in enumerate(emotion_list_ori):
            vid_em_path = os.path.join(vid_path, "neutral", f"{em}_cross_id_dcl_author_manipilator", "faces_aligned")
            new_path = os.path.join(root_gen, emotion_list_ori[idx], vid)
            os.makedirs(new_path, exist_ok=True)
            os.system(f"cp {vid_em_path}/*.png {new_path}")
    
    
    
    ### 2 exp/cmp_crossID/baseline/real
    # root_real = "/data2/JM/code/NED-main/exp/cmp_crossID/real"
    # root_ori = "/data2/JM/code/NED-main/render_test"
    # vid_list = sorted(os.listdir(root_ori))
    # emotion_list = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    # for vid in vid_list:
    #     vid_path = os.path.join(root_ori, vid)
    #     for em in emotion_list:
    #         vid_em_path = os.path.join(vid_path, em, "faces_aligned")
    #         new_path = os.path.join(root_real, em, vid)
    #         os.makedirs(new_path, exist_ok=True)
    #         os.system(f"cp {vid_em_path}/*.png {new_path}")
            
            
            
    # with open("/data2/JM/code/NED-main/video_list.txt", "r") as file:
    #     txt = file.readlines()
    # rm_list = [i.strip("\n") for i in txt]

    # root = "/data2/JM/code/NED-main/train_render_dataset"
    # emotion_list = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    # idx = 0
    # vid_list = ["M003", "M009", "W029", "M012", "M030", "W015"]
    # for vid in vid_list:
    #     vid_path = os.path.join(root, vid, "video", "front")
    #     ii = 0
    #     for em in emotion_list:
    #         if em != "neutral":
    #             vid_em_path = os.path.join(vid_path, em, "level_3")
    #         else:
    #             vid_em_path = os.path.join(vid_path, em, "level_1")
    #         video_list = []
    #         video_list = sorted(os.listdir(vid_em_path))
    #         if em != "neutral":
    #             for _ in range(3):
    #                 video_list.remove(rm_list[idx].split("_")[1])
    #                 idx += 1
    #         else:
    #             for _ in range(4):
    #                 video_list.remove(rm_list[idx].split("_")[1])
    #                 idx += 1
    #         for video in tqdm(video_list):
    #             mv_path = os.path.join(vid_em_path, video)
    #             new_path = os.path.join("/data2/JM/code/NED-main/render_train", vid, "videos")
    #             os.makedirs(new_path, exist_ok=True)
    #             os.system(f"cp {mv_path} {new_path}/{ii}.mp4")
    #             ii += 1
            
    
    
    
    # root = "/data2/JM/code/NED-main/render_test"
    # vid_list = sorted(os.listdir(root))
    # emotion_list = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
    
    # for vid in vid_list:
    #     vid_path = os.path.join(root, vid)
    #     for em in emotion_list:
    #         vid_em_path = os.path.join(vid_path, em)
    #         os.system(f"sh ./preprocess.sh {vid_em_path} test")

        