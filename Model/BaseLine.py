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
from plate_data_loader import *
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


if __name__ == '__main__':
    # load csv
    header = ['track_id', 'image_path', 'lp', 'train']
    
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '..//Picture//2017-IWT4S-CarsReId_LP-dataset//trainVal.csv')

    data_transform = transforms.Compose([transforms.Resize((50,140))])
    
    # train_data = datasets.ImageFolder(train_dir, transform=data_transform)
    
    train_data = Dataloader(filename, transform=data_transform, datasetType = 2, one_hot = True)

    print('Num training images: ', len(train_data))
    batch_size = 100
    num_workers = 0
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                              num_workers=num_workers, shuffle=True)
    # obtain one batch of training images
    dataiter = iter(train_loader)
    # print (train_loader)
    dataiter.next()
    
    images, labels = dataiter.next()
    images = images.numpy() # convert images to numpy for display
    clf = RandomForestRegressor(n_estimators = 10)
    print(labels)
    print(labels.shape)
    print(images.shape)
    new_img=images[:,:,:,0:20]
    print(new_img.shape)
    batch,color, nx, ny = images.shape
    images2=images.reshape((batch, color*nx*ny))
    batch, lx, ly = labels.shape
    labels2=labels.reshape((batch, lx*ly))
    print(labels2)

    clf.fit(images2,labels2)
    pred=clf.predict(images2)
    print(pred)
    sep_pre=pred.reshape((batch, lx,ly))
    my_array = np.zeros([batch, lx])
    my_array2 = np.zeros([batch, lx])
    for i in range(batch) :
        for k in range(lx):
            my_array[i][k]=np.argmax(sep_pre[i][k])
            my_array2[i][k]=np.argmax(labels[i][k])
    print (my_array)
    print (my_array2)
   # print(np.argmax(pred.reshape((batch, lx,ly))))
    errors=abs(my_array2 - my_array)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    #print(accuracy_score(my_array2,my_array))
    
    images, labels = dataiter.next()
    batch,color, nx, ny = images.shape
    images2=images.reshape((batch, color*nx*ny))
    batch, lx, ly = labels.shape
    labels2=labels.reshape((batch, lx*ly))
    print(labels2)

    pred=clf.predict(images2)
    print(pred)
    sep_pre=pred.reshape((batch, lx,ly))
    my_array = np.zeros([batch, lx])
    my_array2 = np.zeros([batch, lx])
    for i in range(batch) :
        for k in range(lx):
            my_array[i][k]=np.argmax(sep_pre[i][k])
            my_array2[i][k]=np.argmax(labels[i][k])
    print (my_array)
    print (my_array2)
   # print(np.argmax(pred.reshape((batch, lx,ly))))
    errors=abs(my_array2 - my_array)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    # plot the images in the batch, along with the corresponding labels
    count = 0
    while(1 == 1):
        if (count==10): break
        #print(labels[count])

        #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        #cv2.imshow('image', np.transpose(images[count], (1, 2, 0)))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        count+=1


    # Train