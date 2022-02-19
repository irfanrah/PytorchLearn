
import torch
import torchvision
from torch. utils. data import Dataset, DataLoader
import numpy as np
import math




class ImageDataset(Dataset):
    def __init__(self):
        self.X1 = np.random.rand(10, 100,100, 3)
        self.X1 = self.X1.astype(np.float32)
        self.X2 = np.random.rand(10, 100,100, 3)
        self.X2 = self.X2.astype(np.float32)
        self.Y = np.random.rand(10, 500)
        self.Y = self.Y.astype(np.float32)

        self.X1 =torch.from_numpy(self.X1) 
        self.X2 =torch.from_numpy(self.X2)
        self.Y = torch.from_numpy(self.Y)
        self.length = self.X1.shape[0]

        
    def __getitem__(self, index):
        return self.X1[index] , self.X2[index] , self.Y[index]
        
    def __len__(self):
        return self.length
        
#dataset = ImageDataset()
#firstdata = dataset[0]
#a ,b ,c = firstdata
#print(a.shape,b.shape,c.shape)