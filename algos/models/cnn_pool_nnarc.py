import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.distributions import Categorical


class SQNet(nn.Module):
    
    def __init__(self, num_inputs, num_actions, hidden_size=128):
        super(SQNet, self).__init__()

        #self.l1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.l1 = nn.Linear(200*400*3 + 2, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size*2)
        self.l3 = nn.Linear(hidden_size*2, hidden_size)
        self.l4 = nn.Linear(hidden_size, 1)


    def forward(self, state, action):

        x = torch.cat([state.view(-1, 200*400*3), action.view(-1, 2)], 1)

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x


class Actor(nn.Module):

    def __init__(self, state_size, action_size, conv_channels=6, kernel_size=3, size_1=32, size_2=64, size_3=32):
        super(Actor, self).__init__()

        self.state_size, channels_in = state_size
        self.action_size = action_size


        self.conv1 = nn.Conv2d(channels_in, conv_channels, kernel_size, stride=1)

        self.size_now = self.conv_output_shape(self.state_size) 

        self.pool1 = nn.MaxPool2d(2, 2)

        self.size_now = (int(self.size_now[0]/2), int(self.size_now[1]/2))

        self.conv2 = nn.Conv2d(conv_channels, conv_channels*2, kernel_size)

        self.size_now = self.conv_output_shape(self.size_now)

        self.pool2 = nn.MaxPool2d(2, 2)

        self.size_now = int(self.size_now[0]/2) * int(self.size_now[1]/2) * conv_channels*2

        self.fc1 = nn.Linear(self.size_now, size_1)

        self.fc2 = nn.Linear(size_1, size_2)

        self.fc3 = nn.Linear(size_2, size_3)

        self.mu = nn.Linear(size_3, action_size)

        self.log_std = nn.Linear(size_3, action_size)


    def forward(self, state):

        x = self.pool1(F.relu(self.conv1(state)))

        x = self.pool2(F.relu(self.conv2(x)))

        x = x.reshape(-1, self.size_now)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        mu = self.mu(x)

        log_std = self.log_std(x)

        return mu, log_std



    def conv_output_shape(self, h_w, kernel_size=3, stride=1, pad=0, dilation=1):
        
        #Utility function for computing output of convolutions
        #takes a tuple of (h,w) and returns a tuple of (h,w)
                
        if type(h_w) is not tuple:
            h_w = (h_w, h_w)
        
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        
        if type(stride) is not tuple:
            stride = (stride, stride)
        
        if type(pad) is not tuple:
            pad = (pad, pad)
        
        h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
        w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1
        
        return h, w

