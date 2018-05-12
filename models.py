import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I



    

        

        
    
      

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.bn1 = nn.BatchNorm2d(1)
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)# input(1,224,224) output(32,220,220)
        self.pool = nn.MaxPool2d(2, 2)
        #self.conv1_drop = nn.Dropout(p=0.9) #output(32,110,110)
        
        self.conv2 = nn.Conv2d(32,128,3)# input(32,110,110) output(64,108,108)
        
        #self.conv2_drop = nn.Dropout(p=0.8)# output(64,54,54)
        
        self.conv3 = nn.Conv2d(128,256,3)# input(64,54,54) and output(128,52,52) and
        
        self.conv3_drop = nn.Dropout(p=0.8)#output(128,26,26)
        self.fc1 = nn.Linear(256*26*26,1000)
        self.fc1_drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1000,136)
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.bn1(x)
        
        x = self.pool(F.relu(self.conv1(x)))
        #x = self.conv1_drop(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        #x = self.conv2_drop(x)
        
        x = self.pool(F.relu(self.conv3(x))) 
        x = self.conv3_drop(x)


        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))   
        x = self.fc1_drop(x) 
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

