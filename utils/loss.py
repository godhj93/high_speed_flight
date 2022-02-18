import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import MeanAbsoluteError

class Loss_grad(Loss):
    '''
    Image gradient loss
    Author: H.J. Shin
    Date: 2022.02.18
    '''
    def __init__(self):
        super(Loss_grad, self).__init__()
        self.MSE = MeanAbsoluteError()
    def call(self, y_true, y_hat):
        dy_hat, dx_hat = tf.image.image_gradients(y_hat)
        dy_true, dx_true = tf.image.image_gradients(y_true)

        loss_x = self.MSE(dx_true, dx_hat)
        loss_y = self.MSE(dy_true, dy_hat)
        
        return loss_x + loss_y

class Loss_depth(Loss):
    '''
    Depth absolute error
    Author: H.J. Shin
    Date: 2022.02.18
    '''
    def __init__(self):
        super(Loss_depth, self).__init__()
        self.MSE = MeanAbsoluteError()

    def call(self, y_true, y_hat):
        
        return self.MSE(y_true, y_hat)

class Loss_DSSIM(Loss):
    '''
    Structural Dissimilarity
    https://en.wikipedia.org/wiki/Structural_similarity
    Author: H.J. Shin
    Date: 2022.02.18
    '''
    def __init__(self):
        super(Loss_DSSIM, self).__init__()

    def call(self, y_true, y_hat):

        SSIM = tf.image.ssim(
            y_true, 
            y_hat,
            max_val = 10, #Maximum depth: 10m
            filter_size=11, # default
            filter_sigma=1.5, # default
            k1=0.01, # default
            k2=0.03 # default
        )

        return 0.5 * (1-SSIM)


class Loss_total(Loss):
    '''
    Loss for autoencoder
    Author: H.J. Shin
    Date: 2022.02.18
    '''
    def __init__(self, alpha=0.1):
        super(Loss_total, self).__init__()

        self.l_grad = Loss_grad()
        self.l_depth = Loss_depth()
        self.l_dssim = Loss_DSSIM()
        self.alpha = alpha
    
    def call(self, y_true, y_hat):

        loss = self.alpha*self.l_depth(y_true, y_hat)
        loss += self.l_grad(y_true, y_hat)
        loss += self.l_dssim(y_true, y_hat)
        return loss

        
    



        



