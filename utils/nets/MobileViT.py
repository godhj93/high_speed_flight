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
        





class MobileViT_Block(nn.Module):

    def __init__(self, in_channels, dimension ):
        super(MobileViT_Block, self).__init__()

        self.in_channels = in_channels
        self.dimension = dimension

        h,w = 2,2

        self.P = h*w
        self.conv1_local_rep = nn.Conv2d(in_channels= self.in_channels, out_channels= self.in_channels, kernel_size=3, stride=1, padding='same')
        self.conv2_local_rep = nn.Conv2d(in_channels= self.in_channels, out_channels= self.dimension, kernel_size=1, stride=1, padding='same')

        #Unfold
        #Extract Patches
        self.extract_patches = torch.nn.Unfold(kernel_size=2, stride=2)
            #Flatten Patches

        #Transformer Encoder

        #Fusion
        self.conv1_fusion = nn.Conv2d(in_channels=self.dimension, out_channels=self.in_channels, kernel_size=1,stride=1, padding='same')
        self.conv2_fusion = nn.Conv2d(in_channels=self.in_channels*2, out_channels=self.in_channels, kernel_size=3,stride=1, padding='same')

        self.swish = Swish()
        
    def forward(self, x):

        N,C,H,W = x.shape
        print(x.shape)
        self.fold = torch.nn.Fold(output_size=(H,W), kernel_size=2, stride=2)
        
        #Local reprentations.
        y = self.conv1_local_rep(x)        
        y = self.conv2_local_rep(y)
        print(y.shape) 
        #Transformers as Convolutions
            #Unfold
        y = self.extract_patches(y)
        c,h,w = y.shape
        print(y.shape)        
        y = y.reshape(self.dimension, -1, self.P)
        print(y.shape)        
            #Transformer encoder
        
            #Fold
        y = y.reshape(c,h,w)
        print(y.shape)
        y = self.fold(y)
        print(y.shape)
        y = self.conv1_fusion(y)
        print(y.shape)
        y = torch.cat([x,y],axis=1)
        print(y.shape)
        y = self.conv2_fusion(y)
        print(y.shape)
        return y

    
class Swish(nn.Module):

    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x*self.sigmoid(x)


class Swish(nn.Module):

    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x*self.sigmoid(x)



# Transformer Encoder