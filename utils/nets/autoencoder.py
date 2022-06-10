
from utils.nets.MobileViT import MobileViT
from utils.nets.InvertedMobileViT import InvertedMobileViT
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, input_shape=(1,256,256)):
        super(Autoencoder, self).__init__()

        self.encoder = MobileViT(input_shape=input_shape)
        self.decoder = InvertedMobileViT()       

    def forward(self, x):
        y = self.encoder(x)
        y = self.decoder(y)
        
        return y
