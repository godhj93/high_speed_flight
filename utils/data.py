import torch
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, Compose
class depth_image_dataset(Dataset):

    def __init__(self, annotations_file, transform, debug):

        self.train_ds_list = pd.read_csv(annotations_file,header=None)
        self.train_ds_list.columns = ['x','y']
        
        if debug:
            print("  Debug Mode... Number of data is 10.")
            self.train_ds_list = self.train_ds_list[:10]

        self.transform = transform
        
        
    def __len__(self):
        return len(self.train_ds_list)
    
    def info(self):
        x,y = self.__getitem__(0)
        return [x.shape, y.shape]

    def __getitem__(self, idx):
        
        img_x = read_image(self.train_ds_list.iloc[idx,0])
        img_y = read_image(self.train_ds_list.iloc[idx,1])

        img_x_norm = torch.add(img_x,img_x.min(), alpha=-1)/img_x.max()
        img_y_norm = torch.add(img_y,img_y.min(), alpha=-1)/img_y.max()

        train_x = self.transform(img_x_norm)
        train_y = self.transform(img_y_norm)

        return train_x, train_y
    
def data_generator(debug, batch_size=32, img_size=224):

    dataset =  depth_image_dataset(
            annotations_file='./data/noisy_train_ds.csv',
            transform=Compose(
                [
                Resize((img_size,img_size)),
                ]),
            debug=debug
            )
    
    train_ds = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=3,
             )
    
    return train_ds, dataset.info()