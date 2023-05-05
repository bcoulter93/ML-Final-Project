# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:14:01 2023

@author: bcoul
"""

# Brian Coulter
# HW 4
# Deep Learning

# Module imports
import json
import csv
import sys
import os
from PIL import Image
import numpy as np
from PIL import ImageFilter
#from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageChops
from PIL import ImageTk
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tvt
import os
import matplotlib.pyplot as plt
from os import listdir
import random
import pandas as pd # module imports
import json
import csv
import sys
import os
from PIL import Image
import numpy as np
from PIL import ImageFilter
#from PIL import ImageFont
from os import listdir
import random
from torch import nn
import torch.nn.functional as F
import seaborn as sns
import math
from natsort import natsorted
from math import sin
from math import pi
import numpy as np
from numpy import arange
from numpy import vstack
from numpy import argmax
from numpy import asarray
from numpy.random import normal
from numpy.random import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot
from random import shuffle
import pickle


# %% Import data
with open('training_data.pickle','rb') as file:
    training_images = pickle.load(file)

with open('testing_data.pickle','rb') as file:
    validation_images = pickle.load(file)

with open('training_ids.pickle','rb') as file:
    training_ids = pickle.load(file)

with open('testing_ids.pickle','rb') as file:
    validation_ids = pickle.load(file)
    
# %% Setup Dataset and dataloader classes
from torch.utils.data import Dataset, DataLoader

class MyDataset(torch.utils.data.Dataset):
    
    def __init__(self,training_set,training_set_ids):  # init statement with single input "rootpath"
        super().__init__()
        self.training_set = training_set
        self.training_set_ids = training_set_ids# store in "self"
        
        # This section extracts list of image names from target folder
        image_list = []  # initialize storage variable
        for images in self.training_set:  # use os to iterate through file names
            image_list.append(images)  # store images names in list
        self.image_list = image_list #store in "self"
        self.length = len(self.image_list)  #store the length of hte list in "self"
        
        my_transforms = tvt.Compose([   # compose set of transforms: NOTE, only using ToTensor here 
            #tvt.ToPILImage(),
            tvt.ToTensor(),
            #tvt.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
            # tvt.RandomAdjustSharpness(0.8,p = 0.5),
            # tvt.ColorJitter(brightness = 0.8, contrast = 0.8, saturation =0.01, hue = 0.02),
            # tvt.GaussianBlur(3,sigma = (0.1,0.2))
            ])
        
        self.my_transforms = my_transforms  # store transforms in "self"
        
    def __len__(self):
        return len(self.image_list) # returpreviously calculated length
    
    def __getitem__(self,index):
        image_for_transform = self.image_list[index] # use os to assemble the path to each image in the dataset  # open a single image at the  indexed path
        image_store = tuple((self.my_transforms(np.array(image_for_transform)),self.training_set_ids[index]))  # store transformed tensors in tuple
            
        return image_store

# %% Build Neural Nets: Only Net1 is used

from torch import nn
import torch.nn.functional as F

#Net 1 Code from the HW document,unaltered
class Net1(nn.Module):
    def __init__( self ):
        super(Net1, self).__init__ ()
        self.conv1 = nn.Conv2d(3, 16 , 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16 , 32 , 3)
        self.fc1 = nn.Linear(6272 , 64) #XXXX = 6272, explained in pdf
        self.fc2 = nn.Linear(64 , 5) #XX = 5 because there are 5 categories

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #print(x)
        return x

# Net 2 is identical to Net1, but includes padding for each of its convolutional layers
class Net2(nn.Module):
    def __init__( self ):
        super(Net2, self).__init__ ()
        self.conv1 = nn.Conv2d(3, 16 , 3, padding =1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16 , 32 , 3,padding =1)
        self.fc1 = nn.Linear(8192 , 64) #XXXX = 8192, explained in pdf
        self.fc2 = nn.Linear(64 , 5) #XX = 5 because there are 5 categories

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #print(x)
        return x

class Net3(nn.Module):
    def __init__( self ):
        super(Net3, self).__init__ ()
        self.conv1 = nn.Conv2d(3,16 , 3, padding =1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16 , 32 , 3,padding =1)
        self.conv3 = nn.Conv2d(32 , 32 , 3,padding =1)
        self.conv4 = nn.Conv2d(32 , 32 , 3,padding =1)
        self.conv5 = nn.Conv2d(32 , 32 , 3,padding =1)
        self.conv6 = nn.Conv2d(32 , 32 , 3,padding =1)
        self.conv7 = nn.Conv2d(32 , 32 , 3,padding =1)
        self.conv8 = nn.Conv2d(32 , 32 , 3,padding =1)
        self.conv9 = nn.Conv2d(32 , 32 , 3,padding =1)
        self.conv10 = nn.Conv2d(32 , 32 , 3,padding =1)
        self.conv11 = nn.Conv2d(32 , 32 , 3,padding =1)
        self.conv12 = nn.Conv2d(32 , 32 , 3,padding =1)# add conv3-12 which has 32 input channels, 32 output channels, and a 3x3 kernel
        self.norm = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(8192 , 64)
        self.fc2 = nn.Linear(64 , 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.norm(F.relu(self.conv3(x)))  # add conv3 in the activation function 10 times
        x = self.norm(F.relu(self.conv4(x)))  # batchnorm is included to get rid of vanishing gradient errors
        x = self.norm(F.relu(self.conv5(x)))
        x = self.norm(F.relu(self.conv6(x)))
        x = self.norm(F.relu(self.conv7(x)))
        x = self.norm(F.relu(self.conv8(x)))
        x = self.norm(F.relu(self.conv9(x)))
        x = self.norm(F.relu(self.conv10(x)))
        x = self.norm(F.relu(self.conv11(x)))
        x = self.norm(F.relu(self.conv12(x)))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# %% Form training and testing loop, instantiate dataloaders
import seaborn as sns
import pandas as pd

training_dataset = MyDataset(training_images,training_ids) 
training_dataloader = DataLoader(training_dataset,batch_size = 15,shuffle = False) # Create training datalaoder
testing_dataset = MyDataset(validation_images,validation_ids)
testing_dataloader = DataLoader(testing_dataset,batch_size = 15,shuffle = False) # Create testing dataloader

# NOTE: this code is setup to run training and validation on my local GPU

def NetRunner(B1,B2,LR): # function to run each net
    net = Net1()
    device = torch.device("cuda:0")  # use local nvidia GPU
    net = net.to(device)      
    criterion = torch.nn.CrossEntropyLoss()
    beta1 = B1# use starting specs from HW documnet
    beta2 = B2
    optimizer = torch.optim.Adam(
    net.parameters(), lr = LR, betas = (beta1,beta2))
    confusion = torch.zeros(5,5).to(device)
    total_evals = 0
    correct_evals = 0 
    
    epochs = 10
    #print('Training') # use setup provided in HW document
    loss_list = []
    epoch_list = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(training_dataloader): # use training dataloader
            running_loss = 0.0
            inputs,labels = data
            inputs,labels = inputs.to(device),labels.to(device) # all needs to be in GPU
            labels = labels.to(torch.int64)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if(i+1) % 100 == 0:
            #     print("[epoch %d, batch % 5d] loss: %.3f" % (epoch + 1, i+1, running_loss / 100))
            #     loss_list.append(running_loss)
            #     epoch_list.append(epoch)

    #print('Testing') 
    test_net = net.eval()
    test_net = test_net.to(device)
    with torch.no_grad():
        for i, data in enumerate(testing_dataloader): # use testing dataloader
            inputs,labels = data
            inputs,labels = inputs.to(device),labels.to(device) # all needs to be in GPU
            tests = test_net(inputs)
            x,predictions = torch.max(tests,1)
            for label,prediction in zip(labels.view(-1),predictions.view(-1)):
                confusion[label.long()][prediction.long()] += 1 # add 1 to indexed block of confusion matrix if combination is found
            total_evals   += labels.size(0)
            correct_evals = (predictions==labels).sum().item()
    
    #Create confusion matrix plot
    plt.figure(figsize = (5,5))
    classes = ['airplane','bus','cat','dog','pizza']
    df = pd.DataFrame(confusion,index = classes,columns=classes).astype(int)
    heatmap = sns.heatmap(df,annot=True,fmt='d') # apply heatmap to data
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(),rotation=0,ha='right',fontsize=15) # apply class names to axes
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(),rotation=45,ha='right',fontsize=15)# apply class names to axes
    total = sum(confusion.flatten())
    correct = confusion[0][0].item() + confusion[1][1].item() + confusion[2][2].item() + confusion[3][3].item() + confusion[4][4].item()
    accuracy = correct/total
    
    return accuracy.item()

# Run for each net
# uncomment to run single cases
#net1_results =  NetRunner(0.808347,0.747796,0.000504082)
#net1_results =  NetRunner(0.92945,0.99000,0.00050)
#net1_results =  NetRunner(0.9,0.99,0.001)

#print(net1_results)

#%% Byesian Optimization functions

def surrogate_function(model, X):
    with catch_warnings():
        simplefilter("ignore")
        return model.predict(X, return_std=True) # fit surrogate function
 
def acquisition_function(X, Xsamples, model):
    y_pred, _ = surrogate_function(model, X) # find best ever score
    best = max(y_pred)
    mu, std = surrogate_function(model, Xsamples) # evaluate surroagte at Xsamples
    probs = norm.cdf((mu - best) / (std+1E-9))# calculate the probability of improvement
    return probs

def run_acquisition(X, y, model,Xsamples):
    scores = acquisition_function(X, Xsamples, model)
    ix = argmax(scores)
    return Xsamples[ix]

# %% NOTE: Due to version incompatibility between sklearn.gaussian process adn scipy,
# I had to edit the GaussianProcessRegressor class to change the max iterations and 
# change the solver method, becasue the default one does not work. 

from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import scipy
from sklearn.utils.optimize import _check_optimize_result

class MyGPR(GaussianProcessRegressor):
    def __init__(self, *args, max_iter=10000000000, gtol=1e-06, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_iter = max_iter
        self._gtol = gtol

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = scipy.optimize.minimize(obj_func, initial_theta, method="slsqp", jac=True, bounds=bounds)
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)
        return theta_opt, func_min
    
# %% Run Bayesian Optimization loop. This calls the BO functions as necessary

print('Beginning Bayesian Optimization Loop: this will take roughly 10 minutes')

X = np.array([[0.9,0.99,0.001],[0.77,0.7,0.003],[0.35,0.8,0.008],[0.12,0.3,0.002],[0.25,0.35,0.002]])
y = np.array([0.5747871994972229,0.5476287007331848,0.4892582297325134,0.5370895862579346,0.5383056402206421]).reshape(5,1)
    
X1samples = np.linspace(0.001,0.99,50).reshape(50,1)
X2samples = np.linspace(0.001,0.99,50).reshape(50,1)
X3samples = np.linspace(0.0001,0.01,50).reshape(50,1)
Xsamples = np.asarray([[i,j,k] for i in X1samples for j in X2samples for k in X3samples]).reshape(125000,3)

y = y.reshape(len(y), 1)
# define the model
length_scale_param=1.0
length_scale_bounds_param=(1e-05, 100000.0)
nu_param= np.inf
matern=Matern(length_scale=length_scale_param,length_scale_bounds=length_scale_bounds_param,nu=nu_param)
model = MyGPR(kernel=matern, n_restarts_optimizer=25,alpha = 1e-10,optimizer = "fmin_l_bfgs_b",normalize_y=False,random_state=None)

model.fit(X, y)
best_list_y = []
best_list_X = []

actual_list = []
# perform the optimization process
for i in range(30):
    x = run_acquisition(X, y, model,Xsamples)
    acc = np.mean(NetRunner(x[0],x[1],x[2]))
    # summarize the finding
    #est, _ = surrogate(model, x.reshape(1,2))
    print(r'b1=%.5f,b2=%.5f,lr=%.5f, accuracy=%.5f' % (x[0],x[1],x[2],acc))
    if i % 1 == 0:
        downsamples = Xsamples[::50]
        mu,std = surrogate_function(model,downsamples)
        X0p, X1p = downsamples[:,0].reshape(50,50), downsamples[:,1].reshape(50,50)
        Z0p = mu.reshape(50,50)
        fig = plt.figure(figsize=(10,8))
        plt.scatter(X[:,0],X[:,1],color = 'r',marker = '*')
        plt.title('Evolution of Best Value')
        plt.xlabel('Function Samples')
        plt.ylabel('Best Accuracy Achieved')
        with catch_warnings():
        # ignore generated warnings
            simplefilter("ignore")
            ax = fig.add_subplot(111)
        ax.pcolormesh(X0p, X1p, Z0p,shading = 'auto')
        
    # add the data to the dataset
    actual_list.append(acc)
    X = vstack((X, x))
    y = vstack((y, [[acc]]))
    # update the model
    model.fit(X, y)
    best_list_y.append(max(y))
    best_list_X.append(X[np.argmax(y)])

ix = np.argmax(y)
print('Best Result: b1=%.5f,b2=%.5f,lr=%.5f, accracy=%.5f' % (X[ix][0],X[ix][1],X[ix][2], y[ix]))

# %%
fig = plt.figure()
plt.title('Evolution of Best Value')
plt.xlabel('Function Samples')
plt.ylabel('Best Accuracy Achieved')
plt.plot(best_list_y)
plt.show()

# %%
