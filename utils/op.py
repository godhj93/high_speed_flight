import torch
from torchsummary import summary
from utils.data import data_generator
from utils.loss import loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

class Trainer:

    def __init__(self, model, epochs, batch_size, img_size, name, DEBUG=False):
        '''
        Args
        '''
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.DEBUG = DEBUG
        self.name = name
        self.train_ds, _ = self.get_dataset()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.loss_fn = loss
        print(f"  Device: {self.device}")
        print(summary(self.model, input_size=(1,self.img_size,self.img_size)))
        self.train_summary_writer = SummaryWriter()
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.LR_Scheduler = StepLR(self.optimizer, step_size=self.epochs/3, gamma=0.1)

    def get_dataset(self):
        return data_generator(batch_size = self.batch_size, img_size = self.img_size)

    def train(self):
        
        pbar = tqdm(self.train_ds)
        self.model.train()

        for e in range(self.epochs):

            loss_avg = []

            for x,y in pbar:
                
                loss = self.train_step(x,y)
                
                loss_avg.append(loss)
                pbar.set_description(f"Epochs: {e+1}/{self.epochs}, Train loss: {np.mean(loss_avg):.4f}, LR: {self.optimizer.param_groups[0]['lr']}")
                self.train_summary_writer.add_scalar('loss/train', loss, e+1)

            self.LR_Scheduler.step()

    def train_step(self, x, y):

        x, y = x.to(self.device), y.to(self.device)

        y_hat = self.model(x)
        loss = self.loss_fn(y, y_hat)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
        

            
    
        

        