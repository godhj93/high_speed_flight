import torch
from utils.data import data_generator
from utils.loss import loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
import os

class Trainer:

    def __init__(self, model, epochs, batch_size, img_size, name, debug=False):
        '''
        Args
        '''
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.DEBUG = debug
        self.train_ds, _ = self.get_dataset()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.loss_fn = loss
        print(f"  Device: {self.device}")
        #gprint(summary(self.model, input_size=(1,self.img_size,self.img_size)))
        
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.LR_Scheduler = StepLR(self.optimizer, step_size=self.epochs/3, gamma=0.1)
        self.PATH = datetime.now().strftime('%m-%d/')
        self.name = name + datetime.now().strftime('_%H%M%S')
        self.train_summary_writer = SummaryWriter(log_dir='logs/' + self.PATH + self.name)

    def get_dataset(self):
        return data_generator(batch_size = self.batch_size, img_size = self.img_size, debug=self.DEBUG)

    def train(self):
        
        self.model.train()

        for e in range(self.epochs):

            loss_avg = []
            pbar = tqdm(self.train_ds)

            for x,y in pbar:
                
                loss = self.train_step(x,y)
                
                loss_avg.append(loss)
                pbar.set_description(f"Epochs: {e+1}/{self.epochs}, Train loss: {np.mean(loss_avg):.4f}, LR: {self.optimizer.param_groups[0]['lr']}")
            self.train_summary_writer.add_scalar('/loss/train', np.mean(loss_avg), e)

            self.LR_Scheduler.step()

        self.save_model()

    def train_step(self, x, y):

        x, y = x.to(self.device), y.to(self.device)

        y_hat = self.model(x)
        loss = self.loss_fn(y, y_hat)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
        
    def save_model(self):
        PATH = './models/' + self.PATH
        isExist = os.path.exists(PATH)
        if not isExist:
            os.makedirs(PATH)        
        torch.save(self.model.state_dict(), os.path.join(PATH, self.name))
        
            
        print(f"Model has been saved in {os.path.join(PATH, self.name)}")



            
    
        

        