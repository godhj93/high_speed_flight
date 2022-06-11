import tensorflow as tf
from tensorflow.keras import layers
from utils.nets.MobileViT import MobileViT
from utils.nets.InvertedMobileViT import InvertedMobileViT
from utils.nets.Binary_DenseNet import BinaryDenseNet
class AutoEncoder(tf.keras.Model):

    def __init__(self, classes, arch='S', size=224):
        super(AutoEncoder, self).__init__()
        self.size = size
        self.encoder = MobileViT(arch=arch, classes=classes).model(input_shape=(self.size,self.size,1))
        
#        self.encoder = BinaryDenseNet(arch='bdn-45', use_binary_downsampling = False, classes= classes).model(input_shape=(self.size,self.size,1))
        print(f"ENCODER: {self.encoder.summary()}")
        self.decoder = InvertedMobileViT().model(input_shape=(3,classes))
        # self.build(input_shape=(None,self.size,self.size,1))
    def call(self, x):

        y = self.encoder(x)
        y = self.decoder(y)

        return y 

    def model(self, input_shape=(224, 224, 1)):
        '''
        This method makes the command "model.summary()" work.
        input_shape: (H,W,C), do not specify batch B
        '''
        x = layers.Input(shape=input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
        
