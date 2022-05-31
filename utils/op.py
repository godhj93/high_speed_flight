import tensorflow as tf
from tqdm import tqdm
from utils.data import data_load
from utils.loss import Loss_total
from tensorflow.keras.optimizers import SGD, Adam
import copy
from datetime import datetime

class Trainer:
    '''
    Train a Autoencoder
    Author: H.J Shin
    Date: 2022.02.17
    '''
    def __init__(self, model, dataset='nyuv2', epochs=50, batch_size= 16, size=256, alpha=0.5, DEBUG=False):
        '''
        model: model for training.
        dataset: nyu, kitti
        epochs: uint
        batch_size: uint
        size: input image size
        DEBUG: debug mode {True, False}
        '''
        super(Trainer, self).__init__()
        self._model = copy.deepcopy(model)
        self._epochs = epochs
        self.train_ds = data_load(batch_size=batch_size, size=size, DEBUG=DEBUG)

        self.train_data_length = self.train_ds.length
        self._batch_size = batch_size

        self._optimizer = Adam(learning_rate = self.LR_Scheduler())
        self.loss = Loss_total(alpha=alpha)
        
        #Tensorboard
        self.time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        train_log_dir = 'logs/gradient_tape/' + self.time + '/train'
        test_log_dir = 'logs/gradient_tape/' + self.time + '/test'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        
        self.train_ds = self.train_ds.get_batched_dataset()
    def LR_Scheduler(self):
        
        self.step = (self.train_data_length)//self._batch_size # Stpes  in one epoch
        print(f"STEP: {self.step}")
        B1 = self.step*(0.5*self._epochs)
        B2 = self.step*(0.75*self._epochs)
        boundaries, values = [B1,B2], [1e-3,1e-4,1e-5]
        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=boundaries, values=values)
        
    def progress_bar(self, dataset):
        if dataset == 'train':
            return tqdm(enumerate(self.train_ds), ncols=0)
        elif dataset == 'test':
            return tqdm(enumerate(self.test_ds), ncols=0)
        else:
            raise ValueError("dataset must be 'train' or 'test'")

    
    def train(self):
        print(f"Initializing...")
        
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)

        for e in range(self._epochs):
            print(f"\nEPOCHS: {e+1}/{self._epochs}")
            
            train_bar = self.progress_bar('train')
            for step,(x,y) in train_bar:
                if step == self.step:
                    break   
                
                self.train_step(x,y)
                train_bar.set_description(f"Loss: {self.train_loss.result().numpy():.4f}")
            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result(), step=e)

            self.reset_metric()
        
        print(f"Training is completed.")
        
    
    def reset_metric(self):

        self.train_loss.reset_states()



    @tf.function
    def train_step(self, x,y):
              
        with tf.GradientTape() as tape:
            y_hat = self._model(x, training=True)
            loss = self.loss(y,y_hat)
        
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
        
        self.train_loss.update_state(loss)
       
    @tf.function
    def test_step(self, x,y):
              
        y_hat = self._model(x, training=False)
        loss = self.loss(y,y_hat)

        self.test_loss.update_state(loss)

    def save_weights(self, name):

        model_path = './models/' + name + '_' + self.time +'.h5'

        self._model.save_weights(model_path)
        print(f'the model weights has been saved in {model_path}')

    def save_model(self, name):

        model_path = './models/' + name + '_' + self.time

        self._model.save(model_path)
        print(f'the model has been saved in {model_path}')


    