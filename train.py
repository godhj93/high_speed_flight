from tkinter import N
import tensorflow as tf
from utils.op import Trainer
import argparse
from utils.nets.AutoEncoder import AutoEncoder

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

parser = argparse.ArgumentParser('Train MobilViT AutoEncoder')
parser.add_argument("--ep", default=10, type=int,help="Epochs")
parser.add_argument("--bs", default=8, type=int,help="Batch Size")
#parser.add_argument("--arch", default='S', type=str,help="Architecture: [S, XS, XSS]")
parser.add_argument("--data", default='NYUv2')
parser.add_argument("--name", default='AutoEncoder')
parser.add_argument("--alpha", type=float, default=0.5)
args = parser.parse_args()

def main():
    
    autoencoder = AutoEncoder(classes=64).model(input_shape=(256,256,1))
    print(autoencoder.summary())
    print(f"Epochs: {args.ep}\nBatch Size: {args.bs}\nAlpha: {float(args.alpha)} \n")
    trainer = Trainer(autoencoder, epochs=args.ep, batch_size=args.bs, size= 256, alpha= args.alpha, DEBUG=False)
    trainer.train()
    trainer.save_model(name=args.name)
    
if __name__ == '__main__':

    main()
    print("Done.")