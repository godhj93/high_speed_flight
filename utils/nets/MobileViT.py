import torch
from torch import nn




class MobileViT(nn.Module):
    
    def __init__(self,arch, input_shape, classes=64):
        super(MobileViT, self).__init__()

        self.arch = arch
        self.input_shape = input_shape

        self.conv1 = nn.Conv2d(in_channels=input_shape[1], out_channels=16, kernel_size=3, stride=2)
        self.zeropad = nn.ZeroPad2d(padding=(1,1,1,1))
        self.MV1 = InvertedRedisual(in_channels=16, out_channels=32, strides=1)

        self.MV2 = InvertedRedisual(in_channels=32, out_channels=64, strides=2)
        self.MV3 = InvertedRedisual(in_channels=64, out_channels=64, strides=1)

        self.MV4 = InvertedRedisual(in_channels=64, out_channels=96, strides=2)
        #MobileViT block

        self.MV5 = InvertedRedisual(in_channels=96, out_channels=128, strides=2)
        #MobileViT block
        
        self.MV6 = InvertedRedisual(in_channels=128, out_channels=160, strides=2)
        #MobileViT block
        self.conv2 = nn.Conv2d(in_channels=160, out_channels=640, kernel_size=1, stride=1)

        self.global_pool = nn.AvgPool2d(kernel_size=7)
        self.logits = nn.Linear(in_features=640, out_features=classes)
        self.relu6 = nn.ReLU6()

    def forward(self, x):

        y = (self.conv1(self.zeropad(x)))
        y = self.MV1(y)
        y = self.MV2(y)
        y = self.MV3(y)
        y = self.MV4(y)
        y = self.MV5(y)
        y = self.MV6(y)
        y = self.conv2(y)
        y = self.global_pool(y)
        y = y.squeeze()
        logits = self.relu6(self.logits(y))
        
        return logits

class InvertedRedisual(nn.Module):

    def __init__(self,in_channels, out_channels, strides=1):
        super(InvertedRedisual, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides=strides

        self.pw_conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels*6, padding='same', stride=1, kernel_size=1)

        if self.strides == 2:
            self.dw_conv1 = nn.Conv2d(in_channels=self.in_channels*6, out_channels=self.in_channels*6, stride=self.strides, kernel_size=3, groups=self.in_channels)
        else:
            self.dw_conv1 = nn.Conv2d(in_channels=self.in_channels*6, out_channels=self.in_channels*6, stride=self.strides, padding='same', kernel_size=3, groups=self.in_channels)

        self.pw_conv2 = nn.Conv2d(in_channels=self.in_channels*6, out_channels=self.out_channels, padding='same', stride=1, kernel_size=1)

        self.relu6 = nn.ReLU6()        
        self.zeropad = nn.ZeroPad2d(padding=(1,1,1,1))
    def forward(self, x):

        y = self.relu6(self.pw_conv1(x))
        if self.strides == 2:
            y = self.zeropad(y)
        y = self.relu6(self.dw_conv1((y)))
        y = self.pw_conv2(y)
        
        if self.strides == 1 and (x.shape == y.shape):
            return torch.add(y, x)
        else:
            return y
        





        
# MobileViT Block

# Transformer Encoder