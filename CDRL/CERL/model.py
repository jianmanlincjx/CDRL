import torch
from torch import nn
from clip import clip
import torch.nn.functional as F
import numpy as np
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

clip

####################################### 7 Classes #######################################
class_names_7 = [
'anger.',
'disgust.', 
'fear.', 
'happiness.',
'neutral.',
'sadness',
'surprise.',
]

class_names_with_context_7 = [
'an expression of anger.',
'an expression of disgust.',
'an expression of fear.',
'an expression of happiness.',
'an expression of neutral.',
'an expression of sadness.',
'an expression of surprise.'
]

# class_descriptor_7 = [
#     'angry',
#     'disgusted',
#     'fear',
#     'happy',
#     'neutral',
#     'sad',
#     'surprised',
# ]


class_descriptor_7 = [
'Intense expression of anger with furrowed eyebrows, narrowed eyes, and a tense demeanor.',
'A disgusted expression featuring a wrinkled nose, lowered eyebrows, and a tightened overall appearance.',
'Fear manifested through raised eyebrows, parted lips, a furrowed brow, and a retracted chin.',
'A joyful display of happiness with a smiling expression, raised cheeks, wrinkled eyes, and arched eyebrows.',
'A neutral demeanor characterized by relaxed facial muscles, a calm expression, a smooth forehead, and unremarkable eyebrows.',
'Sadness conveyed through tears, a downturned expression, drooping upper eyelids, and a wrinkled forehead.',
'Surprise reflected in widened eyes, an open expression, raised eyebrows, and a frozen gaze.',
]




class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        ### prompts: XXXXXXXX ~ 对应的文本embedding [7, 77, 512]
        ### tokenized_prompts： XXXXXXXX ~ 文本对应的token [7, 77]

        x = prompts + self.positional_embedding.type(self.dtype)
        x = prompts
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, class_names, clip_model, contexts_number=8, class_token_position="end"):
        super().__init__()

        n_cls = len(class_names) ## 对情绪的描述，已经提前设定好
        n_ctx = contexts_number ## 待优化的prompt的词数
        dtype = clip_model.dtype

        ctx_dim = clip_model.ln_final.weight.shape[0] # 文本特征的长度

        # random initialization
        ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype) # 7种情绪，每种情绪用八个占位字符表示，每个占位字符的特征维度是512

        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx) ## 待优化的占位字符

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        # class_names = [name.replace("_", " ") for name in class_names]
        name_lens = [len(_tokenizer.encode(name)) for name in class_names]
  
        prompts = [prompt_prefix + " " + name for name in class_names]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()

        ## token 转 embedding
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = class_token_position

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class EmotionMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EmotionMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.mlp(x)


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out
    
class EmotionEmbeddingLearning(nn.Module):
    def __init__(self, input_text, clip_model) -> None:
        super(EmotionEmbeddingLearning, self).__init__()
        
        self.prompt_learner = PromptLearner(input_text, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.image_encoder = clip_model.visual
        self.dtype = clip_model.dtype
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

        self.mlps = nn.ModuleList([EmotionMLP(512, 256, 512) for _ in range(7)])




    def PatchNCELoss(self, f_q, f_k, tau=0.07):
        # batch size, channel size, and number of sample locations
        B, C, S = f_q.shape
        # calculate v * v+: BxSx1
        # np.save("/data2/JM/code/DFER-CLIP/prompt_learning/feature/emotion_512.npy", (f_k * f_q).detach().cpu().numpy())
        # exit()
        l_pos = (f_k * f_q).sum(dim=1)[:, :, None]
        # calculate v * v-: BxSxS
        l_neg = torch.bmm(f_q.transpose(1, 2), f_k)
        # The diagonal entries are not negatives. Remove them.
        identity_matrix = torch.eye(S,dtype=torch.bool)[None, :, :].to(f_q.device)
        l_neg.masked_fill_(identity_matrix, -float('inf'))
        # calculate logits: (B)x(S)x(S+1)
        logits = torch.cat((l_pos, l_neg), dim=2) / tau
        # return PatchNCE loss
        predictions = logits.flatten(0, 1)
        targets = torch.zeros(B * S, dtype=torch.long).cuda()
        return self.cross_entropy_loss(predictions, targets)      

    def forward(self, image):
       ################# Visual Part #################
        B, E, C, H, W = image.shape
        image = image.view(B*E, C, H, W)
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features.view(B, E, -1)
        
        # 对每个情绪特征应用MLP
        mlp_outputs = []
        for i in range(E):
            mlp_output = self.mlps[i](image_features[:, i, :])
            mlp_outputs.append(mlp_output)
        # # 将结果重新调整为原始形状
        image_features = torch.stack(mlp_outputs, dim=1).view(B, E, -1)
        image_features = (image_features / image_features.norm(dim=-1, keepdim=True))
        ###############################################
        
        ################## Text Part ##################
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = (text_features / text_features.norm(dim=-1, keepdim=True))
        # torch.save(text_features, "/data2/JM/code/NED-main/prompt_learning/feature/emotion_embedding_features.pt")
        # exit()
        text_features = text_features.repeat(B, 1, 1)
        ###############################################
        cosine_similarity_pos = torch.mean(F.cosine_similarity(text_features, image_features, dim=1))
        sorted_image_features = torch.flip(image_features, dims=[1])
        cosine_similarity_neg = torch.mean(F.cosine_similarity(text_features, sorted_image_features, dim=1))
        text_features = text_features.permute(0, 2, 1)
        image_features = image_features.permute(0, 2, 1)
        loss = self.PatchNCELoss(text_features, image_features)

        return loss, cosine_similarity_pos, cosine_similarity_neg
    
# from torch.utils.data import DataLoader
# import sys
# import os
# sys.path.append("/data2/JM/code/DFER-CLIP")
# from prompt_learning.dataloader import MEADPairDataloader


# if __name__ == "__main__":
#     train_dataset = MEADPairDataloader()
#     dataloader = DataLoader(train_dataset)
#     CLIP_model, _ = clip.load("ViT-B/32", device='cuda')

#     prompt_learning = EmotionEmbeddingLearning(class_descriptor_7, CLIP_model).cuda()

#     for name, param in prompt_learning.named_parameters():
#         param.requires_grad = False
#     for name, param in prompt_learning.named_parameters():
#         if "prompt_learner" in name:  
#             param.requires_grad = True

#     # 假设你的网络是 prompt_learning
#     for name, param in prompt_learning.named_parameters():
#         print(f"Parameter name: {name}, Requires gradient: {param.requires_grad}")
#     total_params = sum(p.numel() for p in prompt_learning.parameters() if p.requires_grad)
#     print(f"Total trainable parameters: {total_params}")
#     total_params = sum(p.numel() for p in prompt_learning.parameters())
#     print(f"Total parameters: {total_params}")
#     prompt_learning = prompt_learning.cuda()

#     for data in dataloader:
#         data = data.cuda()

#         x = prompt_learning(data)
#         exit()
