import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, block_size, use_1x1conv,stride=1):
        super().__init__()
        self.out_channels=block_size[-1][-1]
        if use_1x1conv:
            self.conv1x1 = nn.LazyConv2d(self.out_channels,kernel_size=1, stride=stride)
        else:
            self.conv1x1=None
        self.module_sequence = nn.Sequential()
        for i, conv_size in enumerate(block_size):
            kernel_size, out_channels = conv_size
            padding_size = (kernel_size-1)//2
            if i==0:
                self.module_sequence.add_module(nn.LazyConv2d(out_channels, kernel_size = kernel_size, padding=padding_size, stride=stride))
            else:
                self.module_sequence.add_module(nn.LazyConv2d(out_channels, kernel_size = kernel_size, padding=padding_size))
            self.module_sequence.add_module(nn.LazyBatchNorm2d())
            if i != len(block_size)-1:
                self.module_sequence.add_module(nn.ReLU())
            
                
    def forward(self,X):
        if self.conv1x1:
            identity = self.conv1x1(X)
        else:
            identity = X
        Y=self.module_sequence(X)
        Y+=identity
        return F.relu(Y)

class Res_152(nn.Module):
    def __init__(self):
        super().__init__()
        self.firstblock = nn.Sequential(nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
                                 nn.LazyBatchNorm2d(),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=3,stride=2, padding=1),
                                )
        self.resblk_1= nn.Sequential()
        for i in range(3):
            if i ==0:
                self.resblk_1.add_module(ResBlock(((1,64), (3,64), (1, 256)), True))
            else:
                self.resblk_1.add_module(ResBlock(((1,64), (3,64), (1, 256))))
        self.resblk_2= nn.Sequential()
        for i in range(8):
            if i==0:
                self.resblk_2.add_module(ResBlock(((1,128),(3,128),(1,512)),True, stride=2))
            else:
                self.resblk_2.add_module(ResBlock(((1,128),(3,128),(1,512))))
        self.resblk_3= nn.Sequential()
        for i in range(36):
            if i==0:
                self.resblk_3.add_module(ResBlock(((1,256), (3,256), (1,1024)),True,stride=2))
            else:
                self.resblk_3.add_module(ResBlock(((1,256), (3,256), (1,1024))))
        self.resblk_4=nn.Sequential()
        for i in range(3):
            if i==0:
                self.resblk_4.add_module(ResBlock(((1,512),(3,512),(1,2048)),True, stride=2))
            else:
                self.resblk_4.add_module(ResBlock(((1,512),(3,512),(1,2048))))
        self.lastblock=nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                     nn.Flatten(),
                                     nn.LazyLinear(1000),
                                    )
        

    def forward(self, X):
        X=self.firstblock(X)
        X=self.resblk_1(X)
        X=self.resblk_2(X)
        X=self.resblk_3(X)
        X=self.resblk_4(X)
        X=self.lastblock(X)
        return X

class ResBlock_v2(nn.Module):
    def __init__(self, block_size, use_1x1conv,stride=1):
        super().__init__()
        self.out_channels=block_size[-1][-1]
        if use_1x1conv:
            self.conv1x1 = nn.LazyConv2d(self.out_channels,kernel_size=1, stride=stride)
        else:
            self.conv1x1=None
        self.module_sequence = nn.Sequential()
        for i, conv_size in enumerate(block_size):
            kernel_size, out_channels = conv_size
            padding_size = (kernel_size-1)//2
            self.module_sequence.add_module(nn.LazyBatchNorm2d())
            self.module_sequence.add_module(nn.ReLU())
            if i==0:
                self.module_sequence.add_module(nn.LazyConv2d(out_channels, kernel_size = kernel_size, padding=padding_size, stride=stride))
            else:
                self.module_sequence.add_module(nn.LazyConv2d(out_channels, kernel_size = kernel_size, padding=padding_size))
    
    
    def forward(self,X):
        if self.conv1x1:
            identity = self.conv1x1(X)
        else:
            identity = X
        Y=self.module_sequence(X)
        Y+=identity 
        return Y
