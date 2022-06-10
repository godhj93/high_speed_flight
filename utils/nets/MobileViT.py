from cmath import log
import torch
from torch import nn
from torch.nn import functional as F

class MobileViT(nn.Module):
    
    def __init__(self, input_shape, classes=64):
        super(MobileViT, self).__init__()

        
        input_channel, _, _ = input_shape
        
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=16, kernel_size=3, stride=2, bias=False)
        self.zeropad = nn.ZeroPad2d(padding=(1,1,1,1))
        self.MV1 = InvertedRedisual(in_channels=16, out_channels=32, strides=1)

        self.MV2 = InvertedRedisual(in_channels=32, out_channels=64, strides=2)
        self.MV3 = InvertedRedisual(in_channels=64, out_channels=64, strides=1)

        self.MV4 = InvertedRedisual(in_channels=64, out_channels=96, strides=2)
        self.ViT1 = MobileViT_Block(in_channels=96, dimension=144, num_of_encoder=2)

        self.MV5 = InvertedRedisual(in_channels=96, out_channels=128, strides=2)
        self.ViT2 = MobileViT_Block(in_channels=128, dimension=192, num_of_encoder=4)
        
        self.MV6 = InvertedRedisual(in_channels=128, out_channels=160, strides=2)
        self.ViT3 = MobileViT_Block(in_channels=160, dimension=240, num_of_encoder=3)

        self.conv2 = nn.Conv2d(in_channels=160, out_channels=640, kernel_size=1, stride=1, bias=False)

        self.global_pool = nn.AvgPool2d(kernel_size=7)
        self.logits = nn.Linear(in_features=640, out_features=classes, bias=False)
        self.relu6 = nn.ReLU6()
        self.swish = Swish()

    def forward(self, x):

        y = self.swish((self.conv1(self.zeropad(x))))
        y = self.MV1(y)
        
        y = self.MV2(y)
        y = self.MV3(y)

        y = self.MV4(y)
        y = self.ViT1(y)
        
        y = self.MV5(y)
        y = self.ViT2(y)

        y = self.MV6(y)
        y = self.ViT3(y)

        y = self.swish(self.conv2(y))
        y = self.global_pool(y)
        
        y = y.squeeze()
        
        logits = self.swish(self.logits(y))
        
        return logits

class InvertedRedisual(nn.Module):

    def __init__(self,in_channels, out_channels, strides=1):
        super(InvertedRedisual, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides=strides
        expand_factor = 4
        self.pw_conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels*expand_factor, padding='same', stride=1, kernel_size=1, bias=False)

        if self.strides == 2:
            self.dw_conv1 = nn.Conv2d(in_channels=self.in_channels*expand_factor, out_channels=self.in_channels*expand_factor, stride=self.strides, kernel_size=3, groups=self.in_channels*expand_factor, bias=False)
        else:
            self.dw_conv1 = nn.Conv2d(in_channels=self.in_channels*expand_factor, out_channels=self.in_channels*expand_factor, stride=self.strides, padding='same', kernel_size=3, groups=self.in_channels*expand_factor, bias=False)

        self.pw_conv2 = nn.Conv2d(in_channels=self.in_channels*expand_factor, out_channels=self.out_channels, padding='same', stride=1, kernel_size=1, bias=False)

        self.relu6 = nn.ReLU6()   
        self.swish = Swish()     
        self.zeropad = nn.ZeroPad2d(padding=(1,1,1,1))

    def forward(self, x):

        y = self.swish(self.pw_conv1(x))
        if self.strides == 2:
            y = self.zeropad(y)
        y = self.swish(self.dw_conv1((y)))
        y = self.pw_conv2(y)
        
        if self.strides == 1 and (x.shape == y.shape):
            return torch.add(y, x)
        else:
            return y
        
class MobileViT_Block(nn.Module):

    def __init__(self, in_channels, dimension, num_of_encoder ):
        super(MobileViT_Block, self).__init__()

        self.in_channels = in_channels
        self.dimension = dimension

        h,w = 2,2

        self.P = h*w
        self.conv1_local_rep = nn.Conv2d(in_channels= self.in_channels, out_channels= self.in_channels, kernel_size=3, stride=1, padding='same', bias=False)
        self.conv2_local_rep = nn.Conv2d(in_channels= self.in_channels, out_channels= self.dimension, kernel_size=1, stride=1, padding='same', bias=False)

        #Unfold
        #Extract Patches
        self.extract_patches = torch.nn.Unfold(kernel_size=2, stride=2)
            

        #Transformer Encoder
        self.encoder = Transformer_Encoder(input_dim=dimension, num_of_encoder=num_of_encoder)
        #Fusion
        self.conv1_fusion = nn.Conv2d(in_channels=self.dimension, out_channels=self.in_channels, kernel_size=1,stride=1, padding='same', bias=False)
        self.conv2_fusion = nn.Conv2d(in_channels=self.in_channels*2, out_channels=self.in_channels, kernel_size=3,stride=1, padding='same', bias=False)

        self.swish = Swish()
        
    def forward(self, x):
        
        N,C,H,W = x.shape
        self.fold = torch.nn.Fold(output_size=(H,W), kernel_size=(2,2), stride=(2,2))
        
        #Local reprentations.
        y = self.swish(self.conv1_local_rep(x))
        y = self.swish(self.conv2_local_rep(y))

        #Transformers as Convolutions
            #Unfold
        y = self.extract_patches(y)
        c,h,w = y.shape
        y = y.reshape(N, -1, self.dimension)
        #y = y.permute(0,3,2,1)

            #Transformer encoder
        y = self.encoder(y)
            #Fold
        y = y.reshape(c,h,w)
        y = self.fold(y)
        y = self.swish(self.conv1_fusion(y))
        y = torch.cat([x,y],axis=1)
        y = self.swish(self.conv2_fusion(y))
        
        return y

    
class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x*self.sigmoid(x)


# Transformer Encoder
class Transformer_Encoder(nn.Module):
    
    def __init__(self, input_dim, num_of_encoder):
        super(Transformer_Encoder, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=input_dim)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_of_encoder)
        
    def forward(self, x):

        return self.encoder(x)

     