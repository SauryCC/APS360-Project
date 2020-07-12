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
import scipy.io
import io
from torch.utils.data import Dataset
import torch
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

# evaluate random forest algorithm for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor

# Reference: https://stackoverflow.com/questions/56696147/pytorch-how-to-create-a-custom-dataset-with-reference-table

class Dataloader_scv(Dataset):
    def __init__(self, csv_path, transform = None, datasetType = 0, one_hot = True):
        """
        Args:
            csv_path (string): path to csv file
            transform: pytorch transforms for transform
            datasetType: Type of data: 0 is training, 1 is validation, 2 is testing
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
        self.datasetType = datasetType
        
        if self.datasetType == 0: # Train
            tmp = sevenLengthPf[sevenLengthPf.iloc[:, 3] == 1]
            self.data_info = tmp.iloc[:int(3*len(tmp)/4), :]

        elif self.datasetType == 1: # Val
            tmp = sevenLengthPf[sevenLengthPf.iloc[:, 3] == 1]
            self.data_info = tmp.iloc[int(3*len(tmp)/4):, :]
        
        elif self.datasetType == 2: # Test
            self.data_info = sevenLengthPf[sevenLengthPf.iloc[:, 3] == 0]

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

class MyNet(nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 48, kernel_size=5, stride=1, padding=2) 
        self.conv2 = nn.Conv2d(48, 64, kernel_size=5, padding=0) 
        self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=0, padding=0) 
        
        self.fc1 = nn.Linear(128 * 18 * 7, 2048) # Fully Connected
        self.fc2 = nn.Linear(2048, 1024) # Fully Connected
        self.fc3 = nn.Linear(1024, 36*7) # ABCDEFGHI = 9 outputs

    def forward(self, x):
        x = self.conv1(x)  # (140-5+2x2)/1 + 1 = 140x50
        
        x = F.relu(x)  # 140x50
        x = self.maxPool1(x) # 70x25
        x = self.conv2(x)  # 70x25
        x = F.relu(x)  # 
        x = self.maxPool2(x) # 35x13
        x = self.conv3(x) # 35x13
        x = F.relu(x)  
        x = self.maxPool2(x) # 18x7

        x = x.view(-1, 128 *18*7)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x
'''
# Code reference: Lab 2, Tut 3b
def evaluate(model, loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, momentum=0.9)
    """ Evaluate the network on the validation set.
     Args:
         model: PyTorch neural network object
         loader: PyTorch data loader for the validation set
         criterion: The loss function
     Returns:
         err: A scalar for the avg classification error over the validation set
         loss: A scalar for the average loss function over the validation set
     """
    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0
    correct = 0
    total = 0 
    count = 0
    for imgs, labels in loader:
        count = count +1
        if use_cuda and torch.cuda.is_available():
          imgs = imgs.cuda()
          labels = labels.cuda()
        print(labels)
        # labels = normalize_label(labels)  # Convert labels to 
        outputs = model(imgs)
        #select index with maximum prediction score
        pred = outputs.max(1, keepdim=True)[1]
        loss = criterion(outputs, labels.float())
        corr = (outputs > 0.0).squeeze().long() != labels
        total_err += int(corr.sum())
        total_loss += loss.item()
        total_epoch += len(labels)
    err = float(total_err) / total_epoch
    loss = float(total_loss) / (count)
    return err, loss
def get_accuracy(model, data_loader):
    correct = 0
    total = 0
    for imgs, labels in data_loader:
        
         
        #############################################
        #To Enable GPU Usage
        if use_cuda and torch.cuda.is_available():
          imgs = imgs.cuda()
          labels = labels.cuda()
        #############################################
        
        output = model(imgs)        
        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total
def train(model, train_loader, val_loader, batch_size=20, learning_rate=0.01, num_epochs=1):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    iters, losses, train_acc, val_acc = [], [], [], []
    # training
    n = 0 # the number of iterations
    start_time=time.time()
    for epoch in range(num_epochs):
        mini_b=0
        mini_batch_correct = 0
        Mini_batch_total = 0
        for imgs, labels in iter(train_loader):
          
            
            #############################################
            #To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
              imgs = imgs.cuda()
              labels = labels.cuda()
            #############################################
            out = model(imgs)             # forward pass
            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch
            ##### Mini_batch Accuracy ##### We don't compute accuracy on the whole trainig set in every iteration!
            pred = out.max(1, keepdim=True)[1]
            mini_batch_correct = pred.eq(labels.view_as(pred)).sum().item()
            Mini_batch_total = imgs.shape[0]
            train_acc.append((mini_batch_correct / Mini_batch_total))
           ###########################
          # save the current training information
            iters.append(n)
            losses.append(float(loss)/batch_size)             # compute *average* loss
            val_acc.append(get_accuracy(model, train=False))  # compute validation accuracy
            n += 1
            mini_b += 1
            print("Iteration: ",n,'Progress: % 6.2f ' % ((epoch * len(train_loader) + mini_b) / (num_epochs * len(train_loader))*100),'%', "Time Elapsed: % 6.2f s " % (time.time()-start_time))
        print ("Epoch %d Finished. " % epoch ,"Time per Epoch: % 6.2f s "% ((time.time()-start_time) / (epoch +1)))
    end_time= time.time()
    # plotting
    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()
    plt.title("Training Curve")
    plt.plot(iters, train_acc, label="Training")
    plt.plot(iters, val_acc, label="Validation")    
    plt.xlabel("Iterations")
    plt.ylabel("Validation Accuracy")
    plt.legend(loc='best')
    plt.show()
    train_acc.append(get_accuracy(model, train=True))
    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))
    print ("Total time:  % 6.2f s  Time per Epoch: % 6.2f s " % ( (end_time-start_time), ((end_time-start_time) / num_epochs) ))
    
'''



    # Train