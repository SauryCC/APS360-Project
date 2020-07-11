import sys
import csv
import numpy as np
import random
import torch.utils.data
import time
import os
import io
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage, Normalize, Compose
from torch.utils.data import DataLoader

# Reference: https://stackoverflow.com/questions/56696147/pytorch-how-to-create-a-custom-dataset-with-reference-table

class Dataloader(Dataset):
    def __init__(self, csv_path, transform = None, test = False, one_hot = True):
        """
        Args:
            csv_path (string): path to csv file
            transform: pytorch transforms for transform
            test: whether to generate train/test loader
            one_hot: whether the label is one-hot list or string of label

        """
        
        # One hot list as label?
        self.one_hot = one_hot

        # Read the csv file
        pf = pd.read_csv(csv_path)

        # Filter the data:

        # Only use 7 digits plates as datasets
        sevenLengthPf = pf[pf.iloc[:, 2].str.len() == 7]

        # Load train/test data
        self.test = test

        if self.test == True:
            self.data_info = sevenLengthPf[sevenLengthPf.iloc[:, 3] == 0]
        else:
            self.data_info = sevenLengthPf[sevenLengthPf.iloc[:, 3] == 1]

        # First column contains the image paths
        self.paths = np.asarray(self.data_info.iloc[:, 1])

        # Second column is the labels
        self.labels = np.asarray(self.data_info.iloc[:, 2])

        # Third column is for an train Boolean
        self.trainBools = np.asarray(self.data_info.iloc[:, 3])

        # Calculate len
        self.data_len = len(self.data_info.index)

        # Transform function
        self.transform = transform

        # Transform to tensor
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):

        # Get image name from the pandas df
        imageName = self.paths[index]
        dirname = os.path.dirname(__file__)
        image_path = os.path.join(dirname, '..//Picture//2017-IWT4S-CarsReId_LP-dataset', imageName)

        # Open image
        img = Image.open(image_path)
        
        # Transform image to tensor
        if self.transform !=None:
            img = self.transform(img)
        imgTensor = self.to_tensor(img)
        
        # Get license plate number
        if(self.one_hot == False):
            label = self.labels[index]
        else:
            # Use one_hot
            # creating initial dataframe
            alphaNumerical_Types = ('0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z')
            listOfPlate = []
            for alphaNumerical in self.labels[index]:
                
                place = alphaNumerical_Types.index(alphaNumerical)
                if place >=0 and place <= 35:
                    # oneHotList = [0] * 36
                    # oneHotList[place] = 1
                    listOfPlate.append(place)
                    
            # import pdb; pdb.set_trace()
            ident = torch.eye(36)
            label = ident[torch.tensor(listOfPlate)]

            # label = listOfPlate

        return (imgTensor, label)

    def __len__(self):
        return self.data_len

if __name__ == '__main__':
    # load csv
    header = ['track_id', 'image_path', 'lp', 'train']
    
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '..//Picture//2017-IWT4S-CarsReId_LP-dataset//trainVal.csv')

    data_transform = transforms.Compose([transforms.Resize((50,140))])
    
    # train_data = datasets.ImageFolder(train_dir, transform=data_transform)
    
    train_data = Dataloader(filename, transform=data_transform, test = False, one_hot = True)

    print('Num training images: ', len(train_data))
    batch_size = 128
    num_workers = 0
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                              num_workers=num_workers, shuffle=True)
    # obtain one batch of training images
    dataiter = iter(train_loader)
    # print (train_loader)
    dataiter.next()
    
    images, labels = dataiter.next()
    images = images.numpy() # convert images to numpy for display

    # plot the images in the batch, along with the corresponding labels
    count = 0
    while(1 == 1):
        if (count==10): break
        print(labels[count])

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', np.transpose(images[count], (1, 2, 0)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        count+=1