
import torch
from torch import nn
from torchmetrics.functional import image_gradients
from ignite.metrics import SSIM
from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *

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

    ssim = SSIM(data_range=1.0,
                kernel_size=(11,11),
                sigma=(1.5,1.5),
                k1=0.01,
                k2=0.03)
    ssim.attach(default_evaluator, 'ssim')
    loss_ssim = default_evaluator.run([[y_true,y_hat]])
    return 0.5 * (1.0-loss_ssim.metrics['ssim'])

def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)

def loss(y_true, y_hat):

    loss = 0
    loss += grad_loss(y_true, y_hat)
    loss += depth_loss(y_true, y_hat)
    loss += DSSIM_loss(y_true, y_hat)

    return loss
