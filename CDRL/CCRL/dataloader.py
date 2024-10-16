import os
import torch
import random
import torchvision
import pickle
import cv2
import json
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import sys
sys.path.append("/data2/JM/code/NED-main")



class MEADPairDataloader(Dataset):
    def __init__(self, mode="train") -> None:
        super(MEADPairDataloader).__init__()
        self.size = 224
        self.root = "/data3/JM/MEAD"
        self.emotion_list = ['angry', 'disgusted', 'fear', 'happy', "neutral", 'sad', 'surprised']
        with open("/data0/JM/STCCL/NED-main_CDRL/decoupled_contrastive_learning/aligned_path36.json", "r") as json_file:
            self.data_file = json.load(json_file)
        if mode == "train":
            self.vid_list = sorted(list(self.data_file.keys()))
            self.vid_list.remove("W021")
        else:
            self.vid_list = ["W021"]
        
        self._img_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((self.size, self.size)),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.num_img = self.calculate_length()
        print(f"{mode} 数据集大小为： {self.num_img}")
        
        
    def calculate_length(self):
        len_pair = 0
        for vid in self.vid_list:
            emotion = self.data_file[vid].keys()
            for em in emotion:
                vid_sub = sorted(self.data_file[vid][em])
                for vid_sub_sub in vid_sub:
                    len_pair += len(self.data_file[vid][em][vid_sub_sub][0])
        return len_pair
            
    
    def __len__(self):
        return self.num_img
    
    def __getitem__(self, index):
        ## 获取文件目录如 “M003”
        vid = random.choice(self.vid_list)
        ## 获取情绪标签 
        emotion_list = list(self.data_file[vid].keys())
        ## 随机选择一种情绪
        emn = random.choice(emotion_list)
        ## 获取对应vid 和 emn下的pair目录
        vid_sub = list(self.data_file[vid][emn].keys())
        ## 随机获取pair目录如 “001_001”
        sub = random.choice(vid_sub)
        ## 获取该目录下的文件索引
        vid_len = len(self.data_file[vid][emn][sub][0])
        assert len(self.data_file[vid][emn][sub][0]) == len(self.data_file[vid][emn][sub][1])
        ## 随机选取一个文件
        idx = random.choice(range(0, vid_len))
        ## 索引pair数据对
        source_vid = sub.split("_")[0]
        target_vid = sub.split("_")[1]
        source_img_idx = self.data_file[vid][emn][sub][0][idx]
        target_img_idx = self.data_file[vid][emn][sub][1][idx]
        assist_len = len(os.listdir(os.path.join(self.root, vid, "align_img", emn, target_vid))) - 2
        assist_index = random.randint(0, assist_len)
        ############################################
        ### c1 e1
        source_img = cv2.imread(os.path.join(self.root, vid, "align_img", "neutral", source_vid, str(source_img_idx).zfill(6)+".jpg"))
        source_audio_feature = torch.from_numpy(np.load(os.path.join(self.root, vid, "audio_feature", "neutral", source_vid, str(source_img_idx).zfill(6)+".npy"))).reshape(1, 512)    
        ### c1 e2
        target_img = cv2.imread(os.path.join(self.root, vid, "align_img", emn, target_vid, str(target_img_idx).zfill(6)+".jpg"))
        # target_audio_feature = torch.from_numpy(np.load(os.path.join(self.root, vid, "audio_feature", emn, target_vid, str(target_img_idx).zfill(6)+".npy"))).reshape(1, 512)    
        ### c2 e2
        assist_img = cv2.imread(os.path.join(self.root, vid, "align_img", emn, target_vid, str(assist_index).zfill(6)+".jpg"))
        assist_audio_feature = torch.from_numpy(np.load(os.path.join(self.root, vid, "audio_feature", emn, target_vid, str(assist_index).zfill(6)+".npy"))).reshape(1, 512)    
        ############################################
        source_img = self._img_transform(source_img)
        target_img = self._img_transform(target_img)
        assist_img = self._img_transform(assist_img)
        return {"source_img": source_img,
                "target_img": target_img,
                "assist_img": assist_img,
                "source_audio_feature": source_audio_feature,
                "assist_audio_feature": assist_audio_feature,
                "emotion": self.emotion_list.index(emn)}
        

        


if __name__ == "__main__":
    data = MEADPairDataloader("train")
    dataloader = DataLoader(data, batch_size=128, num_workers=32, shuffle=False)
    for data in dataloader:
        pass
        # print(data["source_img"].shape)
        # print(data["target_img"].shape)
        # print(data["source_audio_feature"].shape)
        # source = data["source_img"].cuda()
        # target = data["target_img"].cuda()
        # loss, pos, neg = model(source, target)
        # print(loss.item(), pos, neg)
