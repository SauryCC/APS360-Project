import sys
import csv
import numpy as np
import random
import torch.utils.data
import time
import os
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage, Normalize, Compose
from torch.utils.data import DataLoader
import numpy as np
import random

import io
from torch.utils.data import Dataset
import torch

# Reference: https://stackoverflow.com/questions/56696147/pytorch-how-to-create-a-custom-dataset-with-reference-table

class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path, transform = None, test = False):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()

        # Read the csv file
        self.data_info = pd.read_csv(csv_path)

        self.test = test

        if self.test == True:
            self.data_info = self.data_info[self.data_info.iloc[:, 3] == 0]
        else:
            self.data_info = self.data_info[self.data_info.iloc[:, 3] == 1]
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])

        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 2])

        # Third column is for an operation indicator
        self.operation_arr = np.asarray(self.data_info.iloc[:, 3])

        # Calculate len
        self.data_len = len(self.data_info.index)

        

        self.transform = transform

    def __getitem__(self, index):

        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        dirname = os.path.dirname(__file__)
        image_path = os.path.join(dirname, '..//Picture//2017-IWT4S-CarsReId_LP-dataset', single_image_name)

        # Open image
        img_as_img = Image.open(image_path)

        # Check if there is an operation
        some_operation = self.operation_arr[index]

        # Transform image to tensor
        if self.transform !=None:
            img_as_img = self.transform(img_as_img)
        img_as_tensor = self.to_tensor(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len

if __name__ == '__main__':
    # load csv
    header = ['track_id', 'image_path', 'lp', 'train']
    
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '..//Picture//2017-IWT4S-CarsReId_LP-dataset//trainVal.csv')

    data_transform = transforms.Compose([transforms.Resize((50,140))])
    
    # train_data = datasets.ImageFolder(train_dir, transform=data_transform)
    
    train_data = CustomDatasetFromImages(filename, transform=data_transform, test = False)

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