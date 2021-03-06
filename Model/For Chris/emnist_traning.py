# -*- coding: utf-8 -*-
"""emnist_training.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/1zOA0BJRrcOszo9kkTx5WIME5Ka7DfW0u
"""

from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt # for plotting
import torch.optim as optim
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from plate_data_loader import *
from opencvPlateGurss import *
from torch.utils.data import TensorDataset, DataLoader

# define a 2-layer artificial neural network
class Emnist_net2(nn.Module):
    def __init__(self):
        super(Emnist_net, self).__init__()
        self.name = "ANN"
        self.layer1 = nn.Linear(28 * 28, 450)
        self.layer2 = nn.Linear(450, 47)

    def forward(self, img):
        flattened = img.view(-1, 28 * 28)
        activation1 = self.layer1(flattened)
        activation1 = F.relu(activation1)
        activation2 = self.layer2(activation1)
        return activation2



# define a 2-layer artificial neural network
class Emnist_net(nn.Module):
    def __init__(self):
        super(Emnist_net, self).__init__()
        self.name = "CNN"
        self.conv1 = nn.Conv2d(1, 5, 5) #input channel 1, output channel 5 24*24*5
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 10, 5)
        self.fc1 = nn.Linear(67*67*5, 2500)
        self.fc1 = nn.Linear(2500, 47)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,67*67*10)
        x = F.relu(self.fc1(x))
        x=self.fc2(x)
        return x



def get_data_loader(batch_size, split):

  torch.manual_seed(1) # set the random seed
  emnist_data = datasets.EMNIST('data', train= True, split = "balanced", download = True,transform=transforms.ToTensor())
#  print(len(emnist_data))
#  count = np.zeros(47)
#  for data, label in emnist_data:
#    count[label]+=1
#  print(np.max(count), np.min(count))
  np.random.seed(1000)
  indice =np.arange(len(emnist_data)) 
  np.random.shuffle(indice)
  split_idx = int(len(emnist_data)*split)
  train_index = indice[:split_idx]
  val_index = indice[split_idx:]
  train_sampler = SubsetRandomSampler(train_index)
  train_loader = torch.utils.data.DataLoader(emnist_data, batch_size=batch_size,   num_workers=0, sampler=train_sampler) 
  val_sampler = SubsetRandomSampler(val_index)
  val_loader = torch.utils.data.DataLoader(emnist_data, batch_size=batch_size,
                                            num_workers=0, sampler=val_sampler)
  testset = datasets.EMNIST('data', train= False, split = "balanced", download = True,transform=transforms.ToTensor())
  # Get the list of indices to sample from
  #test_sampler = SubsetRandomSampler(np.arange(len(emnist_data)))
  test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            num_workers=0)#, sampler = test_sampler
  return train_loader, val_loader, test_loader

tr,v, te = get_data_loader(1, 0.7)
print(len(tr), len(v), len(te))

def get_accuracy(model, data_loader):
    correct = 0
    total = 0
    for imgs, labels in data_loader:
       
        if use_cuda and torch.cuda.is_available():
          imgs = imgs.cuda()
          labels = labels.cuda()

        output = model(imgs)
        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total

def train(model, lr, batch_size, epochs, split = 0.8):
  train_loader, val_loader, test_loader = get_data_loader(batch_size, split)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=lr)
  epoch = []
  train_acc, val_acc, losses = [],[],[]
  for epo in range(epochs):
    for imgs, labels in iter(train_loader):
      if use_cuda and torch.cuda.is_available():
        imgs = imgs.cuda()
        labels = labels.cuda()
        model.cuda()

      out = model(imgs)             # forward pass
      loss = criterion(out, labels) # compute the total loss
      loss.backward()               # backward pass (compute parameter updates)
      optimizer.step()              # make the updates for each parameter
      optimizer.zero_grad()
      losses.append(loss)
    epoch.append(epo)
    train_acc.append(get_accuracy(model, train_loader))
    val_acc.append(get_accuracy(model, val_loader))
    print("epoch:",epo, train_acc[-1], val_acc[-1])
  model_path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(e_net.name,
                                                   batch_size,
                                                   lr,
                                                   epochs)
  torch.save(e_net.state_dict(), model_path)
  plt.title("Training Curve learning rate:{}, epo:{}, batch_size:{}".format(lr, epo, batch_size))
  plt.plot(losses, label="Train")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.show()

  plt.title("Training Curve learning rate:{}, epo:{}, batch_size:{}".format(lr, epo, batch_size))
  plt.plot(epoch, train_acc, label="Train")
  plt.plot(epoch, val_acc, label="Validation")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.legend(loc='best')
  plt.show()


class Dataloader2(Dataset):
    def __init__(self, csv_path, transform = None, datasetType = 0, one_hot = True):
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
    
tmp_x=[]
tmp_y=[]
no_le=0
le7=0
def check(Dataloader2,index):
    count=0
               # Get image name from the pandas df
    imageName = Dataloader2.paths[index]
    dirname = os.path.dirname(__file__)
    image_path = os.path.join(dirname, '..//Picture//2017-IWT4S-CarsReId_LP-dataset', imageName)
    alphaNumerical_Types = ('0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z')
        # Open image
    img = cv2.imread(image_path)
    skip=False
    try:
        outputs, licenseGuess = slicePic(img)
    except:
           # print("An exception occurred")
        skip=True

    if (skip==False):
    # print out images
        if (len(outputs)==0):
            global no_le
            no_le=no_le+1
        elif (len(outputs)<7):
            global le7
            le7=le7+1
        if (len(outputs)==7):
            count=0
            #for image in outputs:
                #print(list( Dataloader2.labels[index])[count])
                #print(alphaNumerical_Types.index(list( Dataloader2.labels[index])[count]))
             #   global tmp_x
              #  global tmp_y

               # tmp_x.append(image)
                #tmp_y.append(alphaNumerical_Types.index(list( Dataloader2.labels[index])[count]))
                #count=count+1
            return 1
        
    return 0
use_cuda = False

#print(torch.cuda.is_available())
#train(e_net,lr = 0.0001, batch_size = 32, epochs = 30)

#!unzip '/content/drive/My Drive/Colab Notebooks/APS360 LAB/project/imgs.zip' -d '/root/datasets'

#data_dir = "/root/datasets"
#data_transform = transforms.Compose([transforms.Resize((28,28)), transforms.Grayscale(num_output_channels=1),
                                  #    transforms.ToTensor()])
#test_set = datasets.ImageFolder(data_dir, transform=data_transform)
#test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                   #            num_workers=0, shuffle=True)


#Below is extract images from opencv
#train_loader, val_loader, test_loader = get_data_loader(32, 0.8)
    # load csv
header = ['track_id', 'image_path', 'lp', 'train']
    
dirname = os.path.dirname(__file__)
data_transform = transforms.Compose([transforms.Resize((50,140))])
filename = os.path.join(dirname, '..//Picture//2017-IWT4S-CarsReId_LP-dataset//trainVal.csv')
train_data = Dataloader2(filename, transform=data_transform, datasetType = 0, one_hot = False)
print(train_data.labels[10])
count=0
#76032
for i in range(10000):
    count=count+ check(train_data,i)
    print("Total loop:",i)
    print(count)
print(count)
print(no_le)
print(le7)
#print(len(tmp_x))
#print((tmp_x[0].shape))
#print(len(tmp_y))
#ent=Emnist_net()
#train(ent, lr, batch_size, 30, split = 0.8)
