import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np

import random


from functools import partial
from dataclasses import dataclass
from collections import OrderedDict



# All of this code is done by Francesco Saverio Zuppichini https://github.com/FrancescoSaverioZuppichini/ResNet
# Licensed is MIT



class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size
        
conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)





class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels =  in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels



class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
        {
            'conv' : nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            'bn' : nn.BatchNorm2d(self.expanded_channels)
            
        })) if self.should_apply_shortcut else None
        
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels



from collections import OrderedDict
def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs), 
                          'bn': nn.BatchNorm2d(out_channels) }))




class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )



class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
           conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             activation(),
             conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
             activation(),
             conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )
    



class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x



class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], divider=2, deepths=[2,2,2,2], 
                 activation=nn.ReLU, block=ResNetBasicBlock, *args,**kwargs):
        super().__init__()
        

        blocks_sizes = [block_s//divider for block_s in blocks_sizes]
        self.blocks_sizes = blocks_sizes
        
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation, 
                        block=block,  *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]       
        ])
        
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x



class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


class ResNet(nn.Module):
    
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



class ResNetRLGRU(nn.Module):
    
    def __init__(self, in_channels, action_size, aditional_size, block=ResNetBottleNeckBlock, msg_dim=32, deepths=[3, 4, 6, 3], size_mem=256, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        #self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
        
        self.aditional_size = aditional_size

        # For divider 4 -> 256*10*29 
        # TODO change this to func



        self.lin_state_size = 256*10*29
        self.avg = nn.AdaptiveAvgPool2d((10, 4)) # Done to put low amount of parameters
        self.lin_state_size = 256*10*4 # For final use 16 instead of 4




        self.fc1 = nn.Linear(self.lin_state_size + msg_dim + self.aditional_size, self.lin_state_size//10)
        

        self.hidden = nn.GRUCell(self.lin_state_size//10, size_mem)

        self.mu = nn.Linear(size_mem, action_size)

        self.log_std = nn.Linear(size_mem, action_size)      

        self.msg_dim = 32

        self.msg_lin = nn.Linear(size_mem, msg_dim)   


    def forward(self, x_aditional_hidden, msg_in=None):

        x, aditional, hidden = x_aditional_hidden



        #x, aditional = x_aditional_hidden

        #hidden = torch.randn(x.shape[0], 256)

        #import pudb; pudb.set_trace()

        aditional_flat = aditional.reshape(-1, self.aditional_size)

        x = self.encoder(x)

        x = self.avg(x)
        

        #x = self.decoder(x)

        x = x.reshape(-1, self.lin_state_size)


        if(x.shape[0] != hidden.shape[0]):
          hidden = hidden[0:x.shape[0]]

        #msg_in_flat = torch.zeros(aditional_flat.shape[0], self.msg_dim).float()

        #if(msg_in != None):
        #    msg_in_flat = msg_in.reshape(-1, self.msg_dim)



        if(msg_in == None):
            msg_in = torch.zeros(hidden.shape[0], self.msg_dim).float()
        else:
            if(msg_in.shape[0] != hidden.shape[0]):
                #print('msg_in_shape: {}, additional_flat: {}'.format(str(msg_in.shape), str(aditional_flat.shape)))
                #msg_in = msg_in.repeat(hidden.shape[0], 1)
                msg_in = torch.zeros(hidden.shape[0], self.msg_dim).float()

        #print(hidden.shape)
       #print(aditional_flat.shape)
        #print(msg_in.shape)

        x_aug = torch.cat((x, aditional_flat, msg_in.to(x.device)), dim=1)


        x_aug = F.relu(self.fc1(x_aug))

        #hidden.detach_()
        hidden = hidden.detach()

        #import pudb; pudb.set_trace()
        hidden = self.hidden(x_aug, hidden)

        #self.last_hidden = hidden

        msg = F.relu(self.msg_lin(hidden))
        

        mu = self.mu(F.relu(hidden))

        log_std = self.log_std(F.relu(hidden))


        msg[:, 0:12] = aditional_flat


        #print('Msg IN: {}, msg_out : {}'.format(msg_in, msg))

        return mu, log_std, hidden, msg


    def forward_for_summary(self, state):


        x = state

        aditional = torch.randn(x.shape[0], 12).cuda()

        hidden = torch.randn(x.shape[0], 256).cuda()


        aditional_flat = aditional.reshape(-1, self.aditional_size)

        x = self.encoder(x)
        
        x = self.avg(x)

        print(x.shape)

        x = x.reshape(-1, self.lin_state_size)

        x_aug = torch.cat((x, aditional_flat), dim=1)


        x_aug = F.relu(self.fc1(x_aug))

        #hidden.detach_()
        hidden = hidden.detach()

        hidden = self.hidden(x_aug, hidden)

        #self.last_hidden = hidden

        mu = self.mu(F.relu(hidden))

        log_std = self.log_std(F.relu(hidden))

        return log_std


class ResNetRLGRUCritic(ResNetRLGRU):
    def __init__(self, in_channels, action_size, aditional_size, block=ResNetBottleNeckBlock, msg_dim=32, deepths=[3, 4, 6, 3], size_mem=256, *args, **kwargs):
        super().__init__(in_channels, action_size, aditional_size, block=ResNetBottleNeckBlock, deepths=[3, 4, 6, 3], size_mem=256, *args, **kwargs)

        self.fc1 = nn.Linear(self.lin_state_size + self.aditional_size + msg_dim + 2, self.lin_state_size//10)

        self.msg_dim = msg_dim

        self.msg_lin = nn.Linear(size_mem, msg_dim) 

        self.val = nn.Linear(size_mem, 1) 

    def forward(self, x_aditional_hidden, action, msg_in=None):

        x, aditional, hidden = x_aditional_hidden

        #x, aditional = x_aditional_hidden

        #hidden = torch.randn(x.shape[0], 256)

        #import pudb; pudb.set_trace()

        aditional_flat = aditional.reshape(-1, self.aditional_size)

        x = self.encoder(x)

        x = self.avg(x)
        

        #x = self.decoder(x)

        x = x.reshape(-1, self.lin_state_size)

        action_flat = action.reshape(-1, 2)

        

        msg_in_flat = torch.zeros(action_flat.shape[0], self.msg_dim).float().to(x.device)

        if(msg_in != None):
            msg_in_flat = msg_in.reshape(-1, self.msg_dim).to(x.device)


        x_aug = torch.cat((x, aditional_flat, action_flat, msg_in_flat), dim=1)


        x_aug = F.relu(self.fc1(x_aug))

        #hidden.detach_()
        hidden = hidden.detach()

        #import pudb; pudb.set_trace()
        hidden = self.hidden(x_aug, hidden)

        #self.last_hidden = hidden

        msg = F.relu(self.msg_lin(hidden))

        val = torch.tanh(self.val(F.relu(hidden)))

        return val, msg

def resnet101(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 4, 6, 3])




if __name__ == "__main__":
    
    from torchsummary import summary
    
    #image_im = torch.randn((4, 3, 300, 900))
    
    obs_imgs = torch.randn((4, 3, 300, 900))
    hiddens = torch.randn((4, 256))
    aditionals = torch.randn((4, 2, 6))

    #in_channels, action_size, aditional_size
    rl_model = ResNetRLGRU(3, 2, 12)
    #model = resnet101(3, 4)
    
    res = rl_model((obs_imgs, aditionals, hiddens))
    print(res)

    #summary(rl_model.cuda(), (3, 300, 900))





""" FINAL PRODUCTION 16
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 150, 450]           4,704
       BatchNorm2d-2         [-1, 32, 150, 450]              64
              ReLU-3         [-1, 32, 150, 450]               0
         MaxPool2d-4          [-1, 32, 75, 225]               0
        Conv2dAuto-5          [-1, 32, 75, 225]           9,216
       BatchNorm2d-6          [-1, 32, 75, 225]              64
              ReLU-7          [-1, 32, 75, 225]               0
        Conv2dAuto-8          [-1, 32, 75, 225]           9,216
       BatchNorm2d-9          [-1, 32, 75, 225]              64
 ResNetBasicBlock-10          [-1, 32, 75, 225]               0
       Conv2dAuto-11          [-1, 32, 75, 225]           9,216
      BatchNorm2d-12          [-1, 32, 75, 225]              64
             ReLU-13          [-1, 32, 75, 225]               0
       Conv2dAuto-14          [-1, 32, 75, 225]           9,216
      BatchNorm2d-15          [-1, 32, 75, 225]              64
 ResNetBasicBlock-16          [-1, 32, 75, 225]               0
      ResNetLayer-17          [-1, 32, 75, 225]               0
           Conv2d-18          [-1, 64, 38, 113]           2,048
      BatchNorm2d-19          [-1, 64, 38, 113]             128
       Conv2dAuto-20          [-1, 64, 38, 113]          18,432
      BatchNorm2d-21          [-1, 64, 38, 113]             128
             ReLU-22          [-1, 64, 38, 113]               0
       Conv2dAuto-23          [-1, 64, 38, 113]          36,864
      BatchNorm2d-24          [-1, 64, 38, 113]             128
 ResNetBasicBlock-25          [-1, 64, 38, 113]               0
       Conv2dAuto-26          [-1, 64, 38, 113]          36,864
      BatchNorm2d-27          [-1, 64, 38, 113]             128
             ReLU-28          [-1, 64, 38, 113]               0
       Conv2dAuto-29          [-1, 64, 38, 113]          36,864
      BatchNorm2d-30          [-1, 64, 38, 113]             128
 ResNetBasicBlock-31          [-1, 64, 38, 113]               0
      ResNetLayer-32          [-1, 64, 38, 113]               0
           Conv2d-33          [-1, 128, 19, 57]           8,192
      BatchNorm2d-34          [-1, 128, 19, 57]             256
       Conv2dAuto-35          [-1, 128, 19, 57]          73,728
      BatchNorm2d-36          [-1, 128, 19, 57]             256
             ReLU-37          [-1, 128, 19, 57]               0
       Conv2dAuto-38          [-1, 128, 19, 57]         147,456
      BatchNorm2d-39          [-1, 128, 19, 57]             256
 ResNetBasicBlock-40          [-1, 128, 19, 57]               0
       Conv2dAuto-41          [-1, 128, 19, 57]         147,456
      BatchNorm2d-42          [-1, 128, 19, 57]             256
             ReLU-43          [-1, 128, 19, 57]               0
       Conv2dAuto-44          [-1, 128, 19, 57]         147,456
      BatchNorm2d-45          [-1, 128, 19, 57]             256
 ResNetBasicBlock-46          [-1, 128, 19, 57]               0
      ResNetLayer-47          [-1, 128, 19, 57]               0
           Conv2d-48          [-1, 256, 10, 29]          32,768
      BatchNorm2d-49          [-1, 256, 10, 29]             512
       Conv2dAuto-50          [-1, 256, 10, 29]         294,912
      BatchNorm2d-51          [-1, 256, 10, 29]             512
             ReLU-52          [-1, 256, 10, 29]               0
       Conv2dAuto-53          [-1, 256, 10, 29]         589,824
      BatchNorm2d-54          [-1, 256, 10, 29]             512
 ResNetBasicBlock-55          [-1, 256, 10, 29]               0
       Conv2dAuto-56          [-1, 256, 10, 29]         589,824
      BatchNorm2d-57          [-1, 256, 10, 29]             512
             ReLU-58          [-1, 256, 10, 29]               0
       Conv2dAuto-59          [-1, 256, 10, 29]         589,824
      BatchNorm2d-60          [-1, 256, 10, 29]             512
 ResNetBasicBlock-61          [-1, 256, 10, 29]               0
      ResNetLayer-62          [-1, 256, 10, 29]               0
    ResNetEncoder-63          [-1, 256, 10, 29]               0
AdaptiveAvgPool2d-64          [-1, 256, 10, 16]               0
           Linear-65                 [-1, 4096]     167,825,408
          GRUCell-66                  [-1, 256]               0
           Linear-67                    [-1, 2]             514
           Linear-68                    [-1, 2]             514
================================================================
Total params: 170,625,316
Trainable params: 170,625,316
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 3.09
Forward/backward pass size (MB): 163.84
Params size (MB): 650.88
Estimated Total Size (MB): 817.81
----------------------------------------------------------------
"""


""" LOCAL TRAINING  4 instead of 16
torch.Size([2, 256, 10, 4])
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 150, 450]           4,704
       BatchNorm2d-2         [-1, 32, 150, 450]              64
              ReLU-3         [-1, 32, 150, 450]               0
         MaxPool2d-4          [-1, 32, 75, 225]               0
        Conv2dAuto-5          [-1, 32, 75, 225]           9,216
       BatchNorm2d-6          [-1, 32, 75, 225]              64
              ReLU-7          [-1, 32, 75, 225]               0
        Conv2dAuto-8          [-1, 32, 75, 225]           9,216
       BatchNorm2d-9          [-1, 32, 75, 225]              64
 ResNetBasicBlock-10          [-1, 32, 75, 225]               0
       Conv2dAuto-11          [-1, 32, 75, 225]           9,216
      BatchNorm2d-12          [-1, 32, 75, 225]              64
             ReLU-13          [-1, 32, 75, 225]               0
       Conv2dAuto-14          [-1, 32, 75, 225]           9,216
      BatchNorm2d-15          [-1, 32, 75, 225]              64
 ResNetBasicBlock-16          [-1, 32, 75, 225]               0
      ResNetLayer-17          [-1, 32, 75, 225]               0
           Conv2d-18          [-1, 64, 38, 113]           2,048
      BatchNorm2d-19          [-1, 64, 38, 113]             128
       Conv2dAuto-20          [-1, 64, 38, 113]          18,432
      BatchNorm2d-21          [-1, 64, 38, 113]             128
             ReLU-22          [-1, 64, 38, 113]               0
       Conv2dAuto-23          [-1, 64, 38, 113]          36,864
      BatchNorm2d-24          [-1, 64, 38, 113]             128
 ResNetBasicBlock-25          [-1, 64, 38, 113]               0
       Conv2dAuto-26          [-1, 64, 38, 113]          36,864
      BatchNorm2d-27          [-1, 64, 38, 113]             128
             ReLU-28          [-1, 64, 38, 113]               0
       Conv2dAuto-29          [-1, 64, 38, 113]          36,864
      BatchNorm2d-30          [-1, 64, 38, 113]             128
 ResNetBasicBlock-31          [-1, 64, 38, 113]               0
      ResNetLayer-32          [-1, 64, 38, 113]               0
           Conv2d-33          [-1, 128, 19, 57]           8,192
      BatchNorm2d-34          [-1, 128, 19, 57]             256
       Conv2dAuto-35          [-1, 128, 19, 57]          73,728
      BatchNorm2d-36          [-1, 128, 19, 57]             256
             ReLU-37          [-1, 128, 19, 57]               0
       Conv2dAuto-38          [-1, 128, 19, 57]         147,456
      BatchNorm2d-39          [-1, 128, 19, 57]             256
 ResNetBasicBlock-40          [-1, 128, 19, 57]               0
       Conv2dAuto-41          [-1, 128, 19, 57]         147,456
      BatchNorm2d-42          [-1, 128, 19, 57]             256
             ReLU-43          [-1, 128, 19, 57]               0
       Conv2dAuto-44          [-1, 128, 19, 57]         147,456
      BatchNorm2d-45          [-1, 128, 19, 57]             256
 ResNetBasicBlock-46          [-1, 128, 19, 57]               0
      ResNetLayer-47          [-1, 128, 19, 57]               0
           Conv2d-48          [-1, 256, 10, 29]          32,768
      BatchNorm2d-49          [-1, 256, 10, 29]             512
       Conv2dAuto-50          [-1, 256, 10, 29]         294,912
      BatchNorm2d-51          [-1, 256, 10, 29]             512
             ReLU-52          [-1, 256, 10, 29]               0
       Conv2dAuto-53          [-1, 256, 10, 29]         589,824
      BatchNorm2d-54          [-1, 256, 10, 29]             512
 ResNetBasicBlock-55          [-1, 256, 10, 29]               0
       Conv2dAuto-56          [-1, 256, 10, 29]         589,824
      BatchNorm2d-57          [-1, 256, 10, 29]             512
             ReLU-58          [-1, 256, 10, 29]               0
       Conv2dAuto-59          [-1, 256, 10, 29]         589,824
      BatchNorm2d-60          [-1, 256, 10, 29]             512
 ResNetBasicBlock-61          [-1, 256, 10, 29]               0
      ResNetLayer-62          [-1, 256, 10, 29]               0
    ResNetEncoder-63          [-1, 256, 10, 29]               0
AdaptiveAvgPool2d-64           [-1, 256, 10, 4]               0
           Linear-65                 [-1, 1024]      10,499,072
          GRUCell-66                  [-1, 256]               0
           Linear-67                    [-1, 2]             514
           Linear-68                    [-1, 2]             514
================================================================
Total params: 13,298,980
Trainable params: 13,298,980
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 3.09
Forward/backward pass size (MB): 163.58
Params size (MB): 50.73
Estimated Total Size (MB): 217.40
----------------------------------------------------------------
"""