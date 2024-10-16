import torch
import torch.nn as nn
import clip
import torch.nn.functional as F

class EmotionSupervision(nn.Module):
    def __init__(self) -> None:
        super(EmotionSupervision, self).__init__()
        CLIP_model, _ = clip.load("ViT-B/32", device='cuda')
        # 将模型参数和缓冲区转换为全精度
        CLIP_model = CLIP_model.to(torch.float32)
        self.image_encoder = CLIP_model.visual
        self.emoiton_semantics_embedding = torch.load("/data2/JM/code/NED-main/prompt_learning/feature/emotion_embedding_features.pt")

    def forward(self, image_ref, image_target, label_index):

        refenerce = self.image_encoder(image_ref).reshape(len(label_index), -1, 512)
        target = self.image_encoder(image_target).reshape(len(label_index), -1, 512)
        
        emotion_sematics_embedding = self.emoiton_semantics_embedding[label_index].reshape(len(label_index), 1, 512).repeat(1, 10, 1)

        emotion_reference = refenerce * emotion_sematics_embedding
        emotion_taregt = target * emotion_sematics_embedding

        loss = torch.mean(1 - F.cosine_similarity(emotion_reference, emotion_taregt, dim=-1))

        return loss



EmotionSupervision()