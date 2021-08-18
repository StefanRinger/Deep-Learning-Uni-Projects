import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """ Conv2D, BatchNorm, ReLU, Conv2D, BatchNorm + skip(input), ReLU + Input"""
    
    
    
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        
        # Define the sub layers of this container layer

        # padding = 1 to keep the height x width dimensions of the feature maps constant.
        # n*n convolution = 2n-1 long -> pad with 1 to get 2n
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(out_channels)
        self.ReLU1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1) # no stride
        self.batchNorm2 = nn.BatchNorm2d(out_channels)
        self.ReLU2 = nn.ReLU(inplace=True)

        # is this here correct:???
        self.skipConv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) # i think stride has to be like in first layer to match the sizes of the feature maps
        self.skipBatchNorm = nn.BatchNorm2d(out_channels)
                
        # I don't know about padding!

        

    def forward(self, x):

        # input -> Conv2D -> BatchNorm -> ReLU -> Conv2D, BatchNorm +=input, ReLU +=Input
        input = x

        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.ReLU1(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        
        # skip connection here?

        input = self.skipConv(input)
        input = self.skipBatchNorm(input)
        x += input

        x = self.ReLU2(x)

        
        return x






class ResNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(ResNet, self).__init__()
        
        ''' the complemte u-net network! '''


        self.netArchitecture = nn.Sequential(
            
            nn.Conv2d(3, 64, 7, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            ResBlock(64, 64, 1),
            ResBlock(64, 128, 2),
            ResBlock(128, 256, 2),
            ResBlock(256, 512, 2),
            nn.AdaptiveAvgPool2d((1,1)),  #GlobalAvgPool(). That is: We want to return the average for every feature map
            nn.Flatten(),
            nn.Linear(512, 2),
            nn.Sigmoid()
        )
        
 

    def forward(self, x):
        
        x = self.netArchitecture(x)
        
        return x