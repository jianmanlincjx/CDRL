import numpy as np
import torch
import torch.nn as nn
import cv2
import sys
import torch
from torch import nn
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    fc_scale = 7 * 7
    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(IResNet, self).__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x, return_features=True):
        out = []
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        return x
    
def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        # 跳跃连接（skip connection）
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        # 添加跳跃连接
        x += self.skip_connection(residual)
        x = self.relu2(x)

        return x
    


class ImageFeatureProcess(nn.Module):
    def __init__(self):
        super(ImageFeatureProcess, self).__init__()
        self.residual_block1 = ResidualBlock(in_channels=256, out_channels=512)
        self.residual_block2 = ResidualBlock(in_channels=512, out_channels=512)
        self.residual_block3 = ResidualBlock(in_channels=512, out_channels=256)
        self.residual_block4 = ResidualBlock(in_channels=256, out_channels=256)

    def forward(self, x):
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.residual_block3(x)
        x = self.residual_block4(x)

        return x

class AudioFeatureProcess(nn.Module):
    def __init__(self):
        super(AudioFeatureProcess, self).__init__()
       # Input size: (B, 1, 512)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 1, kernel_size=3, stride=1, padding=1)
        # self.bn4 = nn.BatchNorm1d(1)
        # Output size after convolution: (B, 256, 512)
        self.fc = nn.Linear(1 * 512, 1 * 28 * 28)

    def forward(self, x):
        # Input: (B, 1, 512)
        x = self.conv1(x)
        # x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        # x = self.bn3(x)
        x = nn.functional.relu(x)
        x = self.conv4(x)
        # x = self.bn4(x)
        x = nn.functional.relu(x)
        # Reshape to (B, 1, 512)
        x = x.view(x.size(0), 1 * 512)
        # Fully connected layer
        x = self.fc(x)
        # Reshape to (B, 1, 28, 28)
        x = x.view(x.size(0), 1, 28, 28)
        return x


class MaskLearning(nn.Module):
    def __init__(self) -> None:
        super(MaskLearning, self).__init__()
        in_features = 28*28
        out_features = 28*28
        self.image_feature_mapping = ImageFeatureProcess()
        self.audio_feature_mapping = AudioFeatureProcess()

        self.linear_q = nn.Linear(in_features, out_features)
        self.linear_k = nn.Linear(in_features, out_features)
        self.linear_v = nn.Linear(in_features, out_features)
 
    def forward(self, img_feature, audio_feature):
        B, C, H, W = img_feature.shape
        ## image_feature_map-（B，C，H，W） audio_feature_map- (B, C, H, W)
        image_feature_map = self.image_feature_mapping(img_feature)
        audio_feature_map = self.audio_feature_mapping(audio_feature)
        
        query = self.linear_q(audio_feature_map.repeat(1, C, 1, 1).view(B, C, -1))
        key = self.linear_k(image_feature_map.view(B, C, -1))
        value = self.linear_v(img_feature.view(B, C, -1))

        content_attn_weights = F.softmax(torch.bmm(query.permute(0, 2, 1), key) / ((H*W) ** 0.5), dim=-1)

        content_information = torch.bmm(value, content_attn_weights.permute(0, 2, 1)).view(B, C, H, W)
        return content_information

        
class DecoupledContrastiveLearning(nn.Module):
    def __init__(self) -> None:
        super(DecoupledContrastiveLearning, self).__init__()
        ## Freeze Part
        self.img_encoder = iresnet50()        
        ## Training Part
        self.mask_learning = MaskLearning()
            
    def forward(self, source_img, target_img, source_audio, assist_img, assist_audio_feature):
        x_f = self.img_encoder(source_img) # c1 e1
        y_f = self.img_encoder(target_img) # c1 e2
        z_f = self.img_encoder(assist_img) # c2 e2

        B, C, H, W = x_f.shape
         
        x_fc = self.mask_learning(x_f, source_audio) # c1 e1 
        y_fc = self.mask_learning(y_f, source_audio) # c1 e2
        z_fc = self.mask_learning(z_f, assist_audio_feature) # c2 e2
        
        positive_content_similarity = torch.clamp(F.cosine_similarity(x_fc.reshape(B, -1), y_fc.reshape(B, -1)), 0, 1) # c1 and c1
        negative_content_similarity = torch.clamp(F.cosine_similarity(x_fc.reshape(B, -1), z_fc.reshape(B, -1)), 0, 1) # c1 and c2

        content_loss = (1 - torch.mean(positive_content_similarity)) + torch.mean(negative_content_similarity)

        loss_all = content_loss 
        return loss_all, content_loss, torch.mean(positive_content_similarity), torch.mean(negative_content_similarity)
    
if __name__ == "__main__":
    x = torch.rand(size=(4, 256, 28, 28)).cuda()
    s = torch.rand(size=(4, 1, 512)).cuda()
    model = MaskLearning().cuda()
    model(x, s)
    
        
    

    
    