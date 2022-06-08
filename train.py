from utils.op import Trainer
from utils.nets.autoencoder import Autoencoder
import argparse
from string import Template

parser = argparse.ArgumentParser('Train Depth Extractor')
parser.add_argument('--ep', default=10, type=int, help='epochs')
parser.add_argument('--bs', default=32, type=int, help='batch size')
parser.add_argument('--size', default=224, type=int, help='input image size')
parser.add_argument('--name', help='file name of the network')
parser.add_argument('--states', default=64, type=int, help='encoder output dimension')
args = parser.parse_args()

def main():

    template = Template("**Parameters**\n  Epochs: ${epochs}\n  Batch size: ${bs}\n  Size: ${size}\n  File Name: ${name}\n  Num of States: ${states}")
    parameters = template.substitute(epochs=args.ep, bs=args.bs, size=args.size, name=args.name, states=args.states)
    print(parameters)

    autoencoder = Autoencoder()
    trainer = Trainer(model = autoencoder, epochs=args.ep, batch_size=args.bs, img_size=args.size, name=args.name)
    trainer.train()

    
if __name__ == '__main__':

    main()
    