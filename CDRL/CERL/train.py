import torch
from torch.utils.data import DataLoader
import sys
sys.path.append("/data0/JM/STCCL/NED-main_CDRL/CDRL/CERL")
from model import EmotionEmbeddingLearning
from dataloader import MEADPairDataloader
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import clip


class_descriptor_7 = [
    'Intense expression of anger with furrowed eyebrows, narrowed eyes, and a tense demeanor.',
    'A disgusted expression featuring a wrinkled nose, lowered eyebrows, and a tightened overall appearance.',
    'Fear manifested through raised eyebrows, parted lips, a furrowed brow, and a retracted chin.',
    'A joyful display of happiness with a smiling expression, raised cheeks, wrinkled eyes, and arched eyebrows.',
    'A neutral demeanor characterized by relaxed facial muscles, a calm expression, a smooth forehead, and unremarkable eyebrows.',
    'Sadness conveyed through tears, a downturned expression, drooping upper eyelids, and a wrinkled forehead.',
    'Surprise reflected in widened eyes, an open expression, raised eyebrows, and a frozen gaze.',
]

def fixed_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark =  True



if __name__ == "__main__":
    fixed_seed()
    log_dir = "/data0/JM/STCCL/NED-main_CDRL/CDRL/CERL/log"
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    train_data = MEADPairDataloader("train")
    test_data = MEADPairDataloader("test")
    CLIP_model, _ = clip.load("ViT-B/32", device='cuda')
    # 将模型参数和缓冲区转换为全精度
    CLIP_model = CLIP_model.to(torch.float32)
    model = EmotionEmbeddingLearning(class_descriptor_7, CLIP_model).cuda()
    # model.load_state_dict(torch.load("/data2/JM/code/DFER-CLIP/prompt_learning/ckpt/18_prompt_small_test.pth"))
    ########## fixed住其他参数，只训练文本特征 ########
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "prompt_learner" in name:  
            param.requires_grad = True
        if "mlps" in name:  
            param.requires_grad = True
    ###############################################
    
    ######### 打印可学习参数 ########
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}, Requires gradient: {param.requires_grad}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    ###############################################
    
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=48)
    test_dataloader = DataLoader(test_data, batch_size=8, shuffle=True, num_workers=8)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[2, 4, 8],
                                                     gamma=0.1)
    data_len_train = len(train_dataloader)
    data_len_test = len(test_dataloader)


    iter = 0
    for epoch in range(100):
        model.train()
        
        train_loss = 0.0
        pos_all = 0.0
        neg_all = 0.0
        
        iter_epoch = 0
        scheduler.step()
        for batch in train_dataloader:
            optimizer.zero_grad()
            loss, pos, neg = model(batch.cuda())
            loss.backward()
            optimizer.step()
            loss_num = loss.item()
            
            train_loss += loss_num
            pos_all += pos.item()
            neg_all += neg.item()
            iter += 1
            iter_epoch += 1
            print(f"epoch: {epoch}  iter: {iter}  train_loss: {loss_num:.10f} pos: {pos: .10f} neg: {neg: .10f} lr: {optimizer.param_groups[0]['lr']}")
            if iter % 5 == 0:
                writer.add_scalar(f"train_iter/total_loss", train_loss/iter_epoch, iter)   
                writer.add_scalar(f"train_iter/pos", pos_all/iter_epoch, iter)   
                writer.add_scalar(f"train_iter/neg", neg_all/iter_epoch, iter)   
        writer.add_scalar(f"train_epoch/total_loss", train_loss/data_len_train, epoch)
        writer.add_scalar(f"train_epoch/pos", pos_all/data_len_train, epoch)
        writer.add_scalar(f"train_epoch/neg", neg_all/data_len_train, epoch)
        if epoch % 1 == 0:
            torch.save(model.state_dict(), f'/data2/JM/code/DFER-CLIP/prompt_learning/ckpt/{epoch}_prompt_small_test.pth')
              
        model.eval()
        test_train_loss = 0.0
        test_pos_all = 0.0
        test_neg_all = 0.0
        with torch.no_grad():  
            for batch in test_dataloader:
                loss, pos, neg = model(batch.cuda())
                loss_num = loss.item()
                test_train_loss += loss_num
                test_pos_all += pos.item()
                test_neg_all += neg.item()
                print(f"epoch: {epoch}  iter: {iter}  test_loss: {loss_num:.3f} pos: {pos:.3f} neg: {neg:.3f}")
            writer.add_scalar(f"test/total_loss", test_train_loss/data_len_test, epoch)
            writer.add_scalar(f"test/pos", test_pos_all/data_len_test, epoch)
            writer.add_scalar(f"test/neg", test_neg_all/data_len_test, epoch)



