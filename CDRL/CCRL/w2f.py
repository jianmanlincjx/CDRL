import numpy as np
import torch 
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from pydub import AudioSegment
import os
from moviepy.editor import VideoFileClip
from tqdm import tqdm

class Audio2Feature(nn.Module):
    def __init__(self, fps) -> None:
        super(Audio2Feature, self).__init__()
        self.overlap = 15
        self.sample_rate_tgt = 16000
        self.window = 128
        self.fps = fps
        self.w2v_model = self.wav_model()   
        self.audio_feature_map = nn.Linear(1024, 512)

    def wav_model(self, device="cuda:0"):
        bundle = getattr(torchaudio.pipelines, 'WAV2VEC2_XLSR53')
        w2v_model = bundle.get_model().to(device) 
        return w2v_model 
    
    def linear_interpolation(self, features, output_len=None):
        features = features.transpose(1, 2)
        output_features = F.interpolate(features, size=output_len, align_corners=True, mode='linear')
        return output_features.transpose(1, 2)
      
    def data_preprocess(self, waveform, sample_rate, temp_start_frame, temp_end_frame):
        stride = sample_rate // self.fps
        waveform_par = waveform[-1, temp_start_frame * stride:temp_end_frame * stride].unsqueeze(dim=0)
        waveform_par = torchaudio.functional.resample(waveform_par, sample_rate, self.sample_rate_tgt)[0]
        return torch.FloatTensor(waveform_par).cuda().unsqueeze(dim=0)

    def waveform2feature(self, audio, len_frames):
        frame_num = len_frames
        audio, lengths = self.w2v_model.feature_extractor(audio, None)
        audio = self.w2v_model.encoder.extract_features(audio, lengths, num_layers=20)[-1]
        audio_feature = self.linear_interpolation(audio,output_len=frame_num)
        hidden_states = self.audio_feature_map(audio_feature)
        return hidden_states
    
    def infer(self, wav_dir, len_frames):
        waveform, sample_rate = torchaudio.load(wav_dir)
        i = 0
        while i < len_frames:
            if i == 0:
                temp_end_frame = min(i + self.window, len_frames)
            else:
                temp_end_frame = min(i + self.window - self.overlap, len_frames)
            temp_start_frame = max(0, i - self.overlap)
            window_len_frames = temp_end_frame - temp_start_frame
            audio = self.data_preprocess(waveform, sample_rate, temp_start_frame, temp_end_frame)
            with torch.no_grad():
                feature = self.waveform2feature(audio=audio, len_frames=window_len_frames)
            feature = feature.cpu().squeeze(0).numpy()
            if i == 0:
                feature_all = feature
            else:
                feature_all = np.concatenate([feature_all, feature[self.overlap:, :]], axis=0)
            i = temp_end_frame
        return feature_all
    

def extract_audio(video_path, save_path):
    # Load the video clip
    video_clip = VideoFileClip(video_path)
    # Extract audio from the video clip
    audio_clip = video_clip.audio
    # Save the audio clip as a WAV file
    audio_clip.write_audiofile(save_path, codec='pcm_s16le')
    # Close the video clip
    video_clip.close()
    
if __name__ == "__main__":
    # ##debug
    # root = os.getcwd()
    # vid_list = sorted(os.listdir(os.getcwd()))
    # vid_list.remove("model_params_me")
    # vid_list.remove("w2f.py")
    # emotion_list = sorted(os.listdir("/data3/JM/MEAD/M003/video/front"))
    # for vid in vid_list:
    #     for emotion in emotion_list:
    #         img_root = os.path.join(root, vid, "align_img", emotion)
    #         audio_root = os.path.join(root, vid, "audio_feature", emotion)   
    #         ll = sorted(os.listdir(img_root))
    #         for i in ll:
    #             img_len = len(os.listdir(os.path.join(img_root, i)))
    #             audio_len = len(os.listdir(os.path.join(audio_root, i)))
    #             # if img_len != audio_len:
    #                 # npy_list = sorted(os.listdir(os.path.join(audio_root, i)))[-1]
    #                 # new_npy = str(int(npy_list.split(".")[0]) + 1).zfill(6) + ".npy"
    #                 # root_path = os.path.join(audio_root, i)
    #                 # os.system(f"cp {root_path}/{npy_list} {root_path}/{new_npy}")
    #                 # print(f"cp {root_path} {npy_list} {new_npy}")
    # video2wav
    root = os.getcwd()
    vid_list = sorted(os.listdir(os.getcwd()))
    vid_list.remove("model_params_me")
    vid_list.remove("w2f.py")
    emotion_list = sorted(os.listdir("/data3/JM/MEAD/M003/video/front"))
    for vid in vid_list:
        audio_vid_root = os.path.join(root, vid, "audio")
        os.makedirs(audio_vid_root, exist_ok=True)
        for emotion in emotion_list:
            audio_vid_emotion = os.path.join(audio_vid_root, emotion)
            os.makedirs(audio_vid_emotion, exist_ok=True)
            if emotion != "neutral":
                video_root = os.path.join(root, vid, "video/front", emotion, "level_3")
                vid_sub_list = sorted(os.listdir(video_root))
            else:
                video_root = os.path.join(root, vid, "video/front", emotion, "level_1")
                vid_sub_list = sorted(os.listdir(video_root))
            for video in tqdm(vid_sub_list):
                video_path = os.path.join(video_root, video)
                save_path = os.path.join(audio_vid_emotion, video.replace("mp4", "wav"))
                extract_audio(video_path, save_path)
    
    fps = 30
    wav2featue_model = Audio2Feature(fps)
    model_params = torch.load("/data3/JM/MEAD/model_params_me/w2v_model_feature_map.ckpt")
    wav2featue_model.load_state_dict(model_params, strict=True)
    wav2featue_model.eval().cuda()


    root = os.getcwd()
    vid_list = sorted(os.listdir(os.getcwd()))
    vid_list.remove("model_params_me")
    vid_list.remove("w2f.py")

    emotion_list = sorted(os.listdir("/data3/JM/MEAD/M003/video/front"))
    for vid in vid_list:
        for emotion in emotion_list:
            audio_root = os.path.join(root, vid, "audio", emotion)
            audio_list = sorted(os.listdir(audio_root))
            for audio in tqdm(audio_list):
                audio_file = os.path.join(audio_root, audio)
                feature_fold = audio_file.replace("audio", "audio_feature").replace(".wav", "")
                img_fold_len = len(sorted(os.listdir(feature_fold.replace("audio_feature", "align_img"))))

                os.makedirs(feature_fold, exist_ok=True)
                sound = AudioSegment.from_file(audio_file)
                sound_time = sound.duration_seconds
                len_frames = int(sound_time * fps)
                if len_frames != img_fold_len:
                    print(f"{len_frames} != {img_fold_len}")
                    len_frames = img_fold_len
                feature_all = wav2featue_model.infer(wav_dir=audio_file, len_frames=len_frames)
                for i in range(feature_all.shape[0]):
                    save_name = str(i).zfill(6) + ".npy"
                    save_path = os.path.join(feature_fold, save_name)
                    np.save(save_path, feature_all[i])