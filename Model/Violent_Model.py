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
from plate_data_loader import Dataloader_scv

from torch.utils.data.sampler import SubsetRandomSampler

class ViolentNet(nn.Module):

    def __init__(self):
        super(ViolentNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 48, kernel_size=5, stride=1, padding=2) 
        self.conv2 = nn.Conv2d(48, 64, kernel_size=5, padding=0) 
        self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=1) 
        
        self.fc1 = nn.Linear(128 * 17 * 5, 2048) # Fully Connected
        self.fc2 = nn.Linear(2048, 1024) # Fully Connected
        self.fc3 = nn.Linear(1024, 36*7) # 36 Possibilities x 7 char

    def forward(self, x):
        x = self.conv1(x)  # (140-5+2x2)/1 + 1 = 140x50
        x = F.relu(x)  # 140x50
        print(x.shape)
        x = self.maxPool1(x) # 70x25
        print(x.shape)
        x = self.conv2(x)  # 66x21
        print(x.shape)
        x = F.relu(x)  # 
        x = self.maxPool2(x) # 34x11
        print(x.shape)
        x = self.conv3(x) # 32x9
        x = F.relu(x)  
        print(x.shape)
        x = self.maxPool2(x) # 17x5
        print(x.shape)
        x = x.view(-1, 128 *17*5)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x

def normalize_label(labels):
    """
    Given a tensor containing 36x7 possible values, normalize this to list of numbers

    Args:
        labels: a 2D tensor containing 7 lists of probability values for each char
    Returns:
        7 Chars
    """
    listOfChar= []
    alphaNumerical_Types = ('0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z')

    for charOneHotArray in labels:
        maxIndex = np.amax(charOneHotArray)
        listOfChar.append(alphaNumerical_Types[maxIndex])
    return listOfChar

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
    totalCorrect = 0
    total = 0 
    count = 0
    for imgs, labels in loader:
        count = count +1
        if use_cuda and torch.cuda.is_available():
          imgs = imgs.cuda()
          labels = labels.cuda()
        print(labels)
        labels = normalize_label(labels)  # Convert labels to alphabets

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
        labels = normalize_label(labels)  # Convert labels to alphabets
        output = model(imgs)        
        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]

        #correct =
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
            #output = Variable(torch.randn(10, 120).float())
            #target = Variable(torch.FloatTensor(10).uniform_(0, 120).long())
            # labels = labels.squeeze(1)
            print(out.shape)
            print(labels.shape)
            labels = labels.view(128,252)
            print(labels.shape)

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



def get_accuracy_test(model):
    correct = 0
    total = 0
    for imgs, labels in test_loader:
        
         
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


    # Train
if __name__ == '__main__':
    with torch.cuda.device(0):
    # device = torch.device('cuda:1')
    # X = X.to(device)
        use_cuda = False

        model = ViolentNet()
        # load csv
        header = ['track_id', 'image_path', 'lp', 'train']
        
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, '..//Picture//2017-IWT4S-CarsReId_LP-dataset//trainVal.csv')
        
        data_transform = transforms.Compose([transforms.Resize((50,140))])
        
        # train_data = datasets.ImageFolder(train_dir, transform=data_transform)
        
        train_data = Dataloader_scv(filename, transform=data_transform, datasetType = 0, one_hot = True)
        val_data = Dataloader_scv(filename, transform=data_transform, datasetType = 1, one_hot = True)
        test_data = Dataloader_scv(filename, transform=data_transform, datasetType = 2, one_hot = True)

        print('Num training images: ', len(train_data))
        batch_size = 128
        num_workers = 0
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                                num_workers=num_workers, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, 
                                                num_workers=num_workers, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
                                                num_workers=num_workers, shuffle=True)
        
        if use_cuda and torch.cuda.is_available():
            model.cuda()
            print('CUDA is available!  Training on GPU ...')
        else:
            print('CUDA is not available.  Training on CPU ...')


        train(model, train_loader, val_loader, batch_size=32, learning_rate=0.01, num_epochs=10)

        get_accuracy_test(model)

'''
train_loader, val_loader, test_loader = get_data_loader(batch_size)

train(model, train_loader, val_loader, batch_size=20, learning_rate=0.01, num_epochs=10)

'''