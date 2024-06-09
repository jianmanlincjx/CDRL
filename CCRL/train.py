import torch
from torch.utils.data import DataLoader
import sys
sys.path.append("/data0/JM/STCCL/NED-main_CDRL/")
from decoupled_contrastive_learning.dataloader import MEADPairDataloader
from decoupled_contrastive_learning.model import DecoupledContrastiveLearning
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os

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
    log_dir = "/data2/JM/code/NED-main/decoupled_contrastive_learning/log/224_2"
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    train_data = MEADPairDataloader("train")
    test_data = MEADPairDataloader("test")
    model = DecoupledContrastiveLearning().cuda()
    model.img_encoder.load_state_dict(torch.load("/data0/JM/STCCL/NED-main_CDRL/decoupled_contrastive_learning/pretrain_model/backbone.pth"))
    # # Freeze the parameters
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "mask_learning" in name:  
            param.requires_grad = True
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}, Requires gradient: {param.requires_grad}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=64)
    test_dataloader = DataLoader(test_data, batch_size=8, shuffle=True, num_workers=8)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    data_len_train = len(train_dataloader)
    data_len_test = len(test_dataloader)

    iter = 0
    for epoch in range(100):
        model.train()
        train_loss = 0.0
        content_loss_all = 0.0
        pc_all = 0.0
        nc_all = 0.0
        iter_epoch = 0

        for batch in train_dataloader:
            source_img = batch['source_img'].cuda()
            target_img = batch['target_img'].cuda()
            assist_img = batch['assist_img'].cuda()
            audio_feature = batch['source_audio_feature'].cuda()
            assist_audio_feature = batch['assist_audio_feature'].cuda()
            optimizer.zero_grad()
            loss, content_loss, pc, nc = model(source_img, target_img, audio_feature, assist_img, assist_audio_feature)
            loss.backward()
            optimizer.step()
            loss_num = loss.item()
            
            train_loss += loss_num
            content_loss_all += content_loss
            pc_all += pc
            nc_all += nc
            
            iter += 1
            iter_epoch += 1
            print(f"epoch: {epoch}  iter: {iter}  train_loss: {loss_num:.3f} content_loss: {content_loss:.3f}  pc: {pc:.3f} nc: {nc:.3f}  Learning Rate: {optimizer.param_groups[0]['lr']}")
            if iter % 100 == 0:
                writer.add_scalar(f"train_iter/total_loss", train_loss/iter_epoch, iter)   
                writer.add_scalar(f"train_iter/content_loss", content_loss_all/iter_epoch, iter)   
                writer.add_scalar(f"train_iter/pos_content", pc_all/iter_epoch, iter)  
                writer.add_scalar(f"train_iter/neg_content", nc_all/iter_epoch, iter)
        writer.add_scalar(f"train_epoch/total_loss", train_loss/data_len_train, epoch)
        writer.add_scalar(f"train_epoch/content_loss", content_loss_all/data_len_train, epoch)
        writer.add_scalar(f"train_epoch/pos_content", pc_all/data_len_train, epoch)  
        writer.add_scalar(f"train_epoch/neg_content", nc_all/data_len_train, epoch)  
  
        if epoch % 2 == 0:
            torch.save(model.state_dict(), f'/data2/JM/code/NED-main/decoupled_contrastive_learning/model_ckpt_sigmoid/{epoch}_DCL.pth')
            
            
        model.eval()
        test_train_loss = 0.0
        test_content_loss_all = 0.0
        test_pc_all = 0.0
        test_nc_all = 0.0
        with torch.no_grad():  
            for batch in test_dataloader:
                source_img = batch['source_img'].cuda()
                target_img = batch['target_img'].cuda()
                assist_img = batch['assist_img'].cuda()
                audio_feature = batch['source_audio_feature'].cuda()
                assist_audio_feature = batch['assist_audio_feature'].cuda()
                loss, content_loss, pc, nc = model(source_img, target_img, audio_feature, assist_img, assist_audio_feature)
                loss_num = loss.item()
                
                test_train_loss += loss_num
                test_content_loss_all += content_loss
                test_pc_all += pc
                test_nc_all += nc
                print(f"epoch: {epoch}  iter: {iter}  test_loss: {loss_num:.3f} content_loss: {content_loss:.3f} pc: {pc:.3f} nc: {nc:.3f}")
            writer.add_scalar(f"test/total_loss", test_train_loss/data_len_test, epoch)
            writer.add_scalar(f"test/content_loss", test_content_loss_all/data_len_test, epoch)
            writer.add_scalar(f"test/pos_content", test_pc_all/data_len_test, epoch)  
            writer.add_scalar(f"test/neg_content", test_nc_all/data_len_test, epoch)  


