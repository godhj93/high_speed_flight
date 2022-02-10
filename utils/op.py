import tensorflow as tf
from tqdm import tqdm
from utils.data import DataLoader
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from datetime import datetime

class Trainer:
    '''
    Train A Neural Network
    Author: H.J Shin
    Date: 2022.02.10
    '''
    def __init__(self, model, dataset='cifar10', epochs=50, batch_size= 16, size=224):
        '''
        model: model for training.
        dataset: NYUv2
        epochs: uint
        batch_size: uint
        '''
        super(Trainer, self).__init__()
        self._model = model
        self._epochs = epochs
        self._batch_size = batch_size
        dl = DataLoader()
        self.train_ds = dl.get_batched_dataset(self._batch_size)
        self.train_data_length = dl.length
        
        self._optimizer = SGD(nesterov=True, momentum=0.9, learning_rate = self.LR_Scheduler())
        self.MSE = tf.keras.losses.MeanSquaredError()

    def LR_Scheduler(self):
        STEPS = (self.train_data_length)//self._batch_size # Stpes  in one epoch
        print(f"STEPS: {STEPS}")
        B1 = STEPS*(0.5*self._epochs)
        B2 = STEPS*(0.75*self._epochs)
        boundaries, values = [B1,B2], [1e-3,1e-4,1e-5]
        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=boundaries, values=values)
        
    def progress_bar(self):

        return tqdm(self.train_ds, ncols=0) 

    
    def train(self):
        print(f"Start Training...")
          
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_accuracy = tf.keras.metrics.MeanSquaredError('train_accuracy')

        # self.test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
        # self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')
      
        for e in range(self._epochs):
            print(f"\n EPOCHS: {e+1}/{self._epochs}")
            train_bar = self.progress_bar()
            for x,y in train_bar:
                print(x.shape, y.shape)
                self.train_step(x,y)
                train_bar.set_description(f"Loss: {self.train_loss.result().numpy():.4f}, Acc: {self.train_accuracy.result().numpy():.4f}")
            
            # for x,y in test_bar:
            #     self.test_step(x,y)
            #     test_bar.set_description(f"Loss: {self.test_loss.result().numpy():.4f}, Acc: {self.test_accuracy.result().numpy():.4f}")
    
    @tf.function
    def train_step(self, x,y):
              
        with tf.GradientTape() as tape:
            y_hat = self._model(x, training=True)
            loss = self.MSE(y,y_hat)
        
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
        
        self.train_accuracy.update_state(y, y_hat)
        self.train_loss.update_state(loss)
       
    # @tf.function
    # def test_step(self, x,y):
              
    #     y_hat = self._model(x, training=False)
    #     loss = self.CrossEntropy(y,y_hat)

    #     self.test_accuracy.update_state(y, y_hat)
    #     self.test_loss.update_state(loss)


def save_model(model):
    now = datetime.now()
    time = now.strftime("%Y-%m-%d_%H-%M-%S")
    model_path = './models/' + 'model_'+ time 

    model.save(model_path)
    print(f'the model has been saved in {model_path}')