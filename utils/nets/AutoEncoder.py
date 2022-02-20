import tensorflow as tf
from tensorflow.keras import layers
from utils.nets.MobileViT import MobileViT
from utils.nets.InvertedMobileViT import InvertedMobileViT

class AutoEncoder(tf.keras.Model):

    def __init__(self, classes):
        super(AutoEncoder, self).__init__()

        self.encoder = MobileViT(arch='S', classes=classes).model(input_shape=(256,256,1))
        self.decoder = InvertedMobileViT().model(input_shape=(classes))
        self.build(input_shape=(None,256,256,1))
    def call(self, x):

        y = self.encoder(x)
        y = self.decoder(y)

        return y 

    def model(self, input_shape=(256,256,3)):
        '''
        This method makes the command "model.summary()" work.
        input_shape: (H,W,C), do not specify batch B
        '''
        x = layers.Input(shape=input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
        
