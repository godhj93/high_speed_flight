import tensorflow as tf
import larq

class BinaryDenseNet:

    def __init__(self, arch='bdn-45', use_binary_downsampling = False, classes= 10):
        '''
        Default setting: BDN-45
        Weight Clip
        '''
        if arch=='bdn-28':
            self.blocks = [6, 6, 6, 5]
            self.reduction = [2.7, 2.7, 2.2]
        elif arch=='bdn-37':
            self.blocks = [6, 8, 12, 6]
            self.reduction = [3.3, 3.3, 4]
        elif arch=='bdn-45':
            self.blocks = [6, 12, 14, 8]
            self.reduction = [2.7, 3.3, 4]

        self.classes = classes
        self.use_binary_downsampling = use_binary_downsampling
        print(f"{arch} has been loaded!")

    def conv_first(self,input_shape):

        return tf.keras.Sequential([
            tf.keras.layers.Input(input_shape),
            #tf.keras.layers.ZeroPadding2D(padding=((3,3),(3,3))),
            tf.keras.layers.Conv2D(filters = 64, kernel_size=7, strides=2, padding='same', kernel_initializer ='he_normal', use_bias = False),
            # larq.layers.QuantConv2D(
            #     filters=64, kernel_size=7, strides=2, padding='same', kernel_initializer ='he_normal', use_bias=False,
            #     kernel_quantizer='ste_sign',
            #     kernel_constraint='weight_clip',
            #     input_quantizer='ste_sign'
            # ),

            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            #tf.keras.layers.ZeroPadding2D(padding=((1,1),(1,1))),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])

    def model(self, input_shape):
        
        net = self.conv_first(input_shape)
        
        growth_rate = 64
        
        net.add(DenseBlock(self.blocks[0], growth_rate))
        net.add(TransitionBlock(reduction = 1/self.reduction[0], binary = self.use_binary_downsampling))
        
        net.add(DenseBlock(self.blocks[1], growth_rate))
        net.add(TransitionBlock(reduction = 1/self.reduction[1], binary = self.use_binary_downsampling))
        
        net.add(DenseBlock(self.blocks[2], growth_rate))
        net.add(TransitionBlock(reduction = 1/self.reduction[2], binary = self.use_binary_downsampling))
        
        net.add(DenseBlock(self.blocks[3], growth_rate))
        
        net.add(tf.keras.layers.BatchNormalization())
        net.add(tf.keras.layers.Activation('relu'))
        net.add(tf.keras.layers.GlobalAveragePooling2D())
        net.add(tf.keras.layers.Dense(self.classes, kernel_initializer='he_normal',activation='softmax'))
        # net.add(larq.layers.QuantDense(
        #     self.classes, kernel_initializer='he_normal',activation='softmax',
        #     kernel_constraint='weight_clip',
        #     kernel_quantizer='ste_sign',
        #     input_quantizer='ste_sign'
            
        # ))

        return net


class ConvBlock(tf.keras.layers.Layer):
    
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        
        # self.conv = tf.keras.layers.Conv2D(num_channels, kernel_size=3, padding='same', use_bias=False)
        self.conv = larq.layers.QuantConv2D(filters=num_channels, kernel_size=3, padding='same', use_bias=False,
                                            input_quantizer='ste_sign',
                                            kernel_quantizer='ste_sign',
                                            kernel_constraint=larq.constraints.WeightClip(1.3),
                                            kernel_initializer='glorot_normal')
        
        # self.listLayers = [self.bn, self.relu, self.conv]
        self.listLayers = [self.bn, self.conv]
        
    def call(self, x):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)
        
        y = tf.keras.layers.concatenate([x,y], axis=-1)
        return y

class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        
        self.listLayers = []
        for _ in range(num_convs):
            self.listLayers.append(ConvBlock(num_channels))
            
    def call(self, x):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x


class TransitionBlock(tf.keras.layers.Layer):
    num_classes = 0
    def __init__(self, reduction = 0.5, binary=False, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        
        TransitionBlock.num_classes += 1
        
        self.binary = binary
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        # self.conv = x = tf.keras.layers.Conv2D(filters = x.shape[-1] * reduction, kernel_size = 1, use_bias=False)
        self.max_pool = tf.keras.layers.AvgPool2D(pool_size = 2, strides=2)
        self.reduction = reduction
    def build(self, input_shape):
        
        print(f"Binarized Downsampling Layer: {self.binary}")
        if self.binary:
            self.conv = larq.layers.QuantConv2D(filters = input_shape[-1] * self.reduction, kernel_size = 1, use_bias=False,
                                                input_quantizer='ste_sign',
                                                kernel_quantizer='ste_sign',
                                                kernel_constraint=larq.constraints.WeightClip(1.3))
        else:    
            self.conv =tf.keras.layers.Conv2D(filters = input_shape[-1] * self.reduction, kernel_size = 1, kernel_initializer = 'he_normal', use_bias=False)
            
        
    def call(self, x):
        
        x = self.batch_norm(x)
        
        x = self.max_pool(x)
           
        if not self.binary:
            x = self.relu(x)
        
        return self.conv(x)