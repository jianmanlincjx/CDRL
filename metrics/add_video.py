import os



if __name__ == "__main__":
    vid_list = ["M003", "M009", "M012", "M030", "W015", "W029"]
    video_list = sorted(os.listdir("/data2/JM/code/NED-main/result_video/cross-id-author"))
    j = 0
    for vid in vid_list:
        audio_list = sorted(os.listdir(f"/data2/JM/code/NED-main/exp/txt/{vid}/audios"))   
        for i, _ in enumerate(audio_list):
            video = video_list[j]
            audio = audio_list[i]
            video_path = f"/data2/JM/code/NED-main/result_video/cross-id-author/{video}"
            audio_path = f"/data2/JM/code/NED-main/exp/txt/{vid}/audios/{audio}"
            os.system(f"ffmpeg -i {video_path} -i {audio_path} -b:v 10M /data2/JM/code/NED-main/result_video/公司/{vid}_{i}.mp4")
            j += 1