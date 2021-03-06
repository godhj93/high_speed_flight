import tensorflow as tf
from tensorflow.keras import layers
from utils.nets.MobileViT import InvertedResidual, MViT_block

class InvertedMobileViT(tf.keras.Model):

    def __init__(self):
        super(InvertedMobileViT, self).__init__()

        self.flatten = layers.Flatten() # 64*3 = 180 + 12 = 192
        self.linear1 = layers.Dense(640, activation=tf.nn.relu) # 640
        self.reshape = layers.Reshape([1,1,-1]) # 1*1*640
        self.upsample1 = layers.UpSampling2D(size=(8,8)) 
        
        self.point_conv1 = layers.Conv2D(filters=160, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu) # 7*7*160
        self.MViT_block_3 = MViT_block(dim=240, n=3, L=3) # 7*7*160
        self.MV5_1 = InvertedResidual(strides=1, filters=160) # 7*7*160
        self.upsample2 = layers.UpSampling2D(size=(2,2)) # 14*14*128

        self.MViT_block_2 = MViT_block(dim=192, n=3, L=4) # 14*14*128
        self.MV4_1 = InvertedResidual(strides=1, filters=128) #14*14*128
        self.upsample3 = layers.UpSampling2D(size=(2,2)) # 28*28*96

        self.MViT_block_1 = MViT_block(dim=144, n=3, L=2) # 28*28*96
        self.MV3_1 = InvertedResidual(strides= 1, filters= 96) # 28*28*96
        self.upsample4 = layers.UpSampling2D(size=(2,2)) # 56*56*64

        self.MV2_3 = InvertedResidual(strides= 1, filters= 64) # 56*56*64
        self.MV2_2 = InvertedResidual(strides= 1, filters= 64) # 56*56*64
        self.MV2_1 = InvertedResidual(strides= 1, filters= 64) # 56*56*64
        self.upsample5 = layers.UpSampling2D(size=(2,2)) # 112*112*32

        self.MV1_1 = InvertedResidual(strides= 1, filters= 32) # 112*112*32
        self.conv3x3 = layers.Conv2D(kernel_size= 3, filters= 1, strides= 1, padding= 'same') # 112*112*1
        self.upsample6 = layers.UpSampling2D(size=(2,2)) # 224*224*1 #When commented 128*128*1
    
    

    def call(self, x):
        y = self.flatten(x)
        y = self.linear1(y)
        y = self.reshape(y)
        y = self.upsample1(y)

        y = self.point_conv1(y)
        y = self.MViT_block_3(y)
        y = self.MV5_1(y)
        y = self.upsample2(y)

        y = self.MViT_block_2(y)
        y = self.MV4_1(y)
        y = self.upsample3(y)

        y = self.MViT_block_1(y)
        y = self.MV3_1(y)
        y = self.upsample4(y)

        y = self.MV2_3(y)
        y = self.MV2_2(y)
        y = self.MV2_1(y)
        y = self.upsample5(y)

        y = self.MV1_1(y)
        y = self.conv3x3(y)
        y = self.upsample6(y) #When commented output shape will be 112*112*1.

        return y


    def model(self, input_shape):
        '''
        This method makes the command "model.summary()" work.
        input_shape: (H,W,C), do not specify batch B
        '''
        x = layers.Input(shape=input_shape)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        print(model.summary())
        return model

class Upsampling(tf.keras.layers.Layer):
    
    def __init__(self):
        super(Upsampling, self).__init__()
        
    def build(self, input_shape):
        
        B,H,W,C = input_shape
        self.transpose_conv2d_1 = layers.Conv2DTranspose(filters=C, kernel_size=3, strides=8, padding='same', use_bias=False)
        # self.transpose_conv2d_2 = layers.Conv2DTranspose(filters=C, kernel_size=3, strides=4, padding='same', use_bias=False)
        # self.transpose_conv2d_3 = layers.Conv2DTranspose(filters=C, kernel_size=3, strides=2, padding='same', use_bias=False)
        
    def call(self, x):
        
        y = self.transpose_conv2d_1(x)
        # y = self.transpose_conv2d_2(y)
        # y = self.transpose_conv2d_3(y)
        return y


class ForwardResidual(tf.keras.layers.Layer):
    '''
    Forward Residual Block
    Author: H.J. Shin
    Date: 2022.02.12
    '''
    def __init__(self,  filters):
        super(ForwardResidual, self).__init__()
        
        self.filters = filters

        self.depth_conv2d = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same', use_bias=False, activation=tf.nn.relu)
        self.point_conv2d = layers.DepthwiseConv2D(filters=1, kernel_size=3, strides=2, padding='same', use_bias=False, activation=tf.nn.relu)
        
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
      
        self.add = layers.Add()
        
    def build(self, input_shape):

        B,H,W,C = input_shape
        self.transpose_conv2d = layers.Conv2DTranspose(filters=C, kernel_size=3, strides=2, padding='same', use_bias=False, activation=tf.nn.relu)

    def get_config(self):
        config = super().get_config()
        config.update({
        
            "filters": self.filters,
        })
        return config

    def call(self, x):

        y = self.bn2(self.depth_conv2d(x))
        y = self.bn3(self.point_conv2d(y))
        y = self.add([x,y])
        return self.bn1(self.transpose_conv2d(y))