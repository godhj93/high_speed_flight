import torch
from torch import nn


# MV2
class InvertedRedisual(nn.Module):

    def __init__(self,in_channels, out_channels, strides=1):
        super(InvertedRedisual, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides=strides

        self.pw_conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels*6, padding='same', stride=1, kernel_size=1)
        if self.strides ==2:
            self.dw_conv1 = nn.Conv2d(in_channels=self.in_channels*6, out_channels=self.in_channels*6, stride=self.strides, kernel_size=3, groups=self.in_channels)
            self.pw_conv2 = nn.Conv2d(in_channels=self.in_channels*6, out_channels=self.out_channels, padding='same', stride=1, kernel_size=1)
        else:
            self.dw_conv1 = nn.Conv2d(in_channels=self.in_channels*6, out_channels=self.in_channels*6, stride=self.strides, padding='same', kernel_size=3, groups=self.in_channels)
            self.pw_conv2 = nn.Conv2d(in_channels=self.in_channels*6, out_channels=self.in_channels, padding='same', stride=1, kernel_size=1)

        self.relu6 = nn.ReLU6()        
        
    def forward(self, x):

        y = self.relu6(self.pw_conv1(x))
        print(y.shape)
        y = self.relu6(self.dw_conv1(y))
        print(y.shape)
        if self.strides == 2:
            return self.pw_conv2(y)

        else:
            y = self.pw_conv2(y)
            return torch.add(y, x)
        





        
# MobileViT Block

# Transformer Encoder