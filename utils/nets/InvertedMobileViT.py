from cmath import log
import torch
from torch import nn
from torch.nn import functional as F
from utils.nets.MobileViT import Swish, MobileViT_Block, InvertedRedisual
class InvertedMobileViT(nn.Module):
    
    def __init__(self, in_features=64):
        super(InvertedMobileViT, self).__init__()


        #64
        #640 1 1
        self.linear = nn.Linear(in_features=in_features, out_features=in_features*10, bias=False)

        #640 8 8
        self.upsampling = nn.Upsample(scale_factor=8)
        #160 8 8
        self.conv1 = nn.Conv2d(in_channels=640, out_channels=160, kernel_size=1, stride=1, bias=False)
        #160 8 8
        self.ViT1 = MobileViT_Block(in_channels=160, dimension=240, num_of_encoder=3)
        #160 8 8
        self.MV1 = InvertedRedisual(in_channels=160, out_channels=128, strides=1)


        #128 16 16
        #upsampleing scale -> 2
        self.ViT2 = MobileViT_Block(in_channels=128, dimension=192, num_of_encoder=4)
        #128 16 16
        self.MV2 = InvertedRedisual(in_channels=128, out_channels=96, strides=1)

        #96 32 32
        #upsampling scale -> 2
        self.ViT3 = MobileViT_Block(in_channels=96, dimension=144, num_of_encoder=2)
        #96 32 32
        self.MV3 = InvertedRedisual(in_channels=96, out_channels=64, strides=1)
        
        #64 64 64
        self.MV4 = InvertedRedisual(in_channels=64, out_channels=64, strides=1)
        #64 64 64
        self.MV5 = InvertedRedisual(in_channels=64, out_channels=32, strides=1)

        #32 128 128
        #upsampling scale -> 2
        self.MV6 = InvertedRedisual(in_channels=32, out_channels=16, strides=1)

        #16 128 128
        #upsampling scale -> 2
        #self.zeropad = nn.ZeroPad2d(padding=(1,1,1,1))
        self.zeropad = nn.ZeroPad2d(padding=(1,1,1,1))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding='same', bias=False)

        self.relu6 = nn.ReLU6()
        self.swish = Swish()
           

    def forward(self, x):

        N, _ = x.shape
        y = self.swish(self.linear(x)) #640
        y = y.reshape(N,-1,1,1) # 1 640 1 1
        y = F.upsample(y, size=(7,7))  #1 640 8 8, // 7,7 -> 224,224
        y = self.swish(self.conv1(y))  #1 160 8 8
        y = self.ViT1(y) #1 160 8 8
        y = self.MV1(y) #1 128 8 8
        
        #128 16 16
        y = F.upsample(y, scale_factor=(2,2)) #1 128 16 16
        y = self.ViT2(y)# 1 128 16 16
        y = self.MV2(y) #1 96 16 16
        
        y = F.upsample(y, scale_factor=(2,2)) #1 96 32 32
        y = self.ViT3(y) #1 96 32 32
        y = self.MV3(y) #1 64 32 32 
        
        
        y = self.MV4(y)#1 64 32 32 
        y = self.MV5(y) #64 64 64
        
        #32 128 128
        y = F.upsample(y, scale_factor=(2,2)) #96 32 32
        y = self.MV6(y)
        
        #16 128 128
        y = F.upsample(y, scale_factor=(4,4)) #96 32 32
        y = self.swish(self.conv2(y))

        return y.squeeze(axis=0)