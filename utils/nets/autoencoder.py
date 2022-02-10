import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large, MobileNetV3Small, DenseNet201

class Autoencoder(tf.keras.Model):

    def __init__(self):
        super(Autoencoder, self).__init__()

        self._encoder = Encoder()
        self._decoder = Decoder()
        print(f"AutoEncoder has been created!")

    def call(self, x):
        
        y = self._encoder(x)
        y = self._decoder(y)

        return y




class Encoder(tf.keras.Model):

    def __init__(self):
        super(Encoder, self).__init__()
        
        self._backbone = MobileNetV3Large(include_top=False, weights='imagenet')
        self.conv1 = tf.keras.layers.Conv2D(filters=3, kernel_size=1, strides=1, use_bias=False)
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.linear = tf.keras.layers.Dense(300)
        
        print(f"encoder has been loaded!")

    def call(self, x):
        x = self.conv1(x)
        x = self._backbone(x)
        
        x= self.global_avg_pool(x)
        x = self.linear(x)
        return x

class Decoder(tf.keras.Model):

    def __init__(self, decode_filters=16):
        super(Decoder, self).__init__()   

        self.reshaper = tf.keras.layers.Reshape((15,20,-1))
        self.conv2 =  tf.keras.layers.Conv2D(filters=decode_filters, kernel_size=1, padding='same', name='conv2')        
        self.up = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.up1 = tf.keras.layers.Conv2D(filters=decode_filters, kernel_size=1, name='up1', activation='relu')
        self.up2 = tf.keras.layers.Conv2D(filters=decode_filters, kernel_size=1, name='up2', activation='relu')
        self.up3 = tf.keras.layers.Conv2D(filters=decode_filters, kernel_size=1, name='up3', activation='relu')
        self.up4 = tf.keras.layers.Conv2D(filters=decode_filters, kernel_size=1, name='up4', activation='relu')     
        self.conv3 = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3')       

        #reshaped_feature = tf.keras.layers.Reshape(feature, (feature.shape[0], 15, 20, -1))
        print(f"decoder has been loaded!")

    def call(self, feature):
        
        reshaped_feature = self.reshaper(feature)
        up0 = self.up(self.conv2(reshaped_feature))
        up1 = self.up(self.up1(up0))
        up2 = self.up(self.up2(up1))
        up3 = self.up(self.up3(up2))
        #up4 = self.up(self.up4(up3))
        
        return self.conv3(up3)




    
 