
import torch
from torch import nn
from torchmetrics.functional import image_gradients
from torchmetrics import StructuralSimilarityIndexMeasure

def grad_loss(y_true, y_hat):

    MSE = nn.L1Loss(reduction='mean')

    dy_true, dx_true = image_gradients(y_true)
    dy_hat, dx_hat = image_gradients(y_hat)

    loss_x = MSE(dx_true, dx_hat)
    loss_y = MSE(dy_true, dy_hat)

    return loss_x + loss_y

def depth_loss(y_true,y_hat):

    MSE = nn.L1Loss(reduction='mean')

    return MSE(y_true,y_hat)

def DSSIM_loss(y_true, y_hat):

    ssim = StructuralSimilarityIndexMeasure()
    
    loss = ssim(y_hat, y_true)
    
    return 0.5 * (1.0-loss)

def loss(y_true, y_hat):

    alpha = 0.5
    loss = grad_loss(y_true, y_hat)
    loss += alpha*depth_loss(y_true, y_hat)
    loss += DSSIM_loss(y_true, y_hat)
    
    return loss
