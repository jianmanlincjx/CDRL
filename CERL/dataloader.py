import os
import torch
import random
import torchvision
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import clip
# https://huggingface.co/papers/



class MEADPairDataloader(Dataset):
    def __init__(self, mode="train") -> None:
        super(MEADPairDataloader).__init__()
        
        self.emotion_list = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
        self.size = 224
        if mode == "train":
            self.image_list_len = 500000
        else:
            self.image_list_len = 640
        self.angry_emotion_list = []
        self.disgusted_emotion_list = []
        self.fear_emotion_list = []
        self.happy_emotion_list = []
        self.neutral_emotion_list = []
        self.sad_emotion_list = []
        self.surprised_emotion_list = []
        
        self.init_emotion_list(mode)

        self._img_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((self.size, self.size)),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return self.image_list_len
    
    def init_emotion_list(self, mode="train"):
        if mode == "train":
            for emotion in self.emotion_list:
                emotion_list_name = f"self.{emotion}_emotion_list"
                with open(f"/data0/JM/STCCL/NED-main_CDRL/CDRL/CERL/emotion_img/{emotion}.txt") as file:
                    temp = file.readlines()
                    temp_list = [i.strip("\n") for i in temp]
                    exec(f"{emotion_list_name} = temp_list")
        if mode == "test":
            for emotion in self.emotion_list:
                emotion_list_name = f"self.{emotion}_emotion_list"
                with open(f"/data0/JM/STCCL/NED-main_CDRL/CDRL/CERL/emotion_img/{emotion}.txt") as file:
                    temp = file.readlines()
                    temp_list = [i.strip("\n") for i in temp]
                    exec(f"{emotion_list_name} = temp_list") 

    def __getitem__(self, index):
        angry_image = self._img_transform(cv2.imread(random.choice(self.angry_emotion_list))).unsqueeze(0)
        disgusted_image = self._img_transform(cv2.imread(random.choice(self.disgusted_emotion_list))).unsqueeze(0)
        fear_image = self._img_transform(cv2.imread(random.choice(self.fear_emotion_list))).unsqueeze(0)
        happy_image = self._img_transform(cv2.imread(random.choice(self.happy_emotion_list))).unsqueeze(0)
        neutral_image = self._img_transform(cv2.imread(random.choice(self.neutral_emotion_list))).unsqueeze(0)
        sad_image = self._img_transform(cv2.imread(random.choice(self.sad_emotion_list))).unsqueeze(0)
        surprised_image = self._img_transform(cv2.imread(random.choice(self.surprised_emotion_list))).unsqueeze(0)
        image_emotion_list = torch.concat([angry_image, disgusted_image, fear_image, happy_image, neutral_image, sad_image, surprised_image], dim=0)

        return image_emotion_list

# if __name__ == "__main__":
#     CLIP_model, _ = clip.load("ViT-B/32", device='cuda')
#     image_encoder = CLIP_model.visual
#     data = MEADPairDataloader("train")
#     dataloader = DataLoader(data)
#     for data in dataloader:
#         data = data.reshape(7, 3, 224, 224).cuda().half()  
#         print(data.shape)
#         x = image_encoder(data)
#         print(x.shape)
#         exit()