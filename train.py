import tensorflow as tf
from utils.op import Trainer
import argparse
from utils.op import save_model
from utils.nets.autoencoder import Encoder, Decoder, Autoencoder

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)


parser = argparse.ArgumentParser('Train Binary Neural Network | Dataset : CIFAR 10')
parser.add_argument("--ep", default=10, type=int,help="Epochs")
parser.add_argument("--bs", default=16, type=int,help="Batch Size")
parser.add_argument("--data", default='nyu2')
args = parser.parse_args()

def main():

    autoencoder = Autoencoder()
    trainer = Trainer(autoencoder, epochs=args.ep, batch_size=args.bs)
    trainer.train()


    save_model(model= autoencoder._encoder, name='encoder')
    save_model(model= autoencoder, name='autoencoder')
    
if __name__ == '__main__':

    main()

    print("Done.")