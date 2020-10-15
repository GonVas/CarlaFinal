import torch
import torch.nn.functional as F
import torch.nn as nn

from torchvision import transforms
from efficientnet_pytorch import EfficientNet



import torch.quantization

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Reduction_A(nn.Module):
    # 35 -> 17
    def __init__(self, in_channels, k, l, m, n):
        super(Reduction_A, self).__init__()
        self.branch_0 = Conv2d(in_channels, n, 3, stride=2, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, k, 1, stride=1, padding=0, bias=False),
            Conv2d(k, l, 3, stride=1, padding=1, bias=False),
            Conv2d(l, m, 3, stride=2, padding=0, bias=False),
        )
        self.branch_2 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        return torch.cat((x0, x1, x2), dim=1) # 17 x 17 x 1024



class Stem(nn.Module):
    def __init__(self, in_channels):
        super(Stem, self).__init__()
        self.conv2d_1a_3x3 = Conv2d(in_channels, 32, 3, stride=2, padding=0, bias=False)

        self.conv2d_2a_3x3 = Conv2d(32, 32, 3, stride=1, padding=0, bias=False)
        self.conv2d_2b_3x3 = Conv2d(32, 64, 3, stride=1, padding=1, bias=False)

        self.mixed_3a_branch_0 = nn.MaxPool2d(3, stride=2, padding=0)
        self.mixed_3a_branch_1 = Conv2d(64, 96, 3, stride=2, padding=0, bias=False)

        self.mixed_4a_branch_0 = nn.Sequential(
            Conv2d(160, 64, 1, stride=1, padding=0, bias=False),
            Conv2d(64, 96, 3, stride=1, padding=0, bias=False),
        )
        self.mixed_4a_branch_1 = nn.Sequential(
            Conv2d(160, 64, 1, stride=1, padding=0, bias=False),
            Conv2d(64, 64, (1, 7), stride=1, padding=(0, 3), bias=False),
            Conv2d(64, 64, (7, 1), stride=1, padding=(3, 0), bias=False),
            Conv2d(64, 96, 3, stride=1, padding=0, bias=False)
        )

        self.mixed_5a_branch_0 = Conv2d(192, 192, 3, stride=2, padding=0, bias=False)
        self.mixed_5a_branch_1 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        x = self.conv2d_1a_3x3(x) # 149 x 149 x 32
        x = self.conv2d_2a_3x3(x) # 147 x 147 x 32
        x = self.conv2d_2b_3x3(x) # 147 x 147 x 64
        x0 = self.mixed_3a_branch_0(x)
        x1 = self.mixed_3a_branch_1(x)
        x = torch.cat((x0, x1), dim=1) # 73 x 73 x 160
        x0 = self.mixed_4a_branch_0(x)
        x1 = self.mixed_4a_branch_1(x)
        x = torch.cat((x0, x1), dim=1) # 71 x 71 x 192
        x0 = self.mixed_5a_branch_0(x)
        x1 = self.mixed_5a_branch_1(x)
        x = torch.cat((x0, x1), dim=1) # 35 x 35 x 384
        return x


class Inception_A(nn.Module):
    def __init__(self, in_channels):
        super(Inception_A, self).__init__()
        self.branch_0 = Conv2d(in_channels, 96, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 64, 1, stride=1, padding=0, bias=False),
            Conv2d(64, 96, 3, stride=1, padding=1, bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv2d(in_channels, 64, 1, stride=1, padding=0, bias=False),
            Conv2d(64, 96, 3, stride=1, padding=1, bias=False),
            Conv2d(96, 96, 3, stride=1, padding=1, bias=False),
        )
        self.brance_3 = nn.Sequential(
            nn.AvgPool2d(3, 1, padding=1, count_include_pad=False),
            Conv2d(384, 96, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.brance_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class Inception_B(nn.Module):
    def __init__(self, in_channels):
        super(Inception_B, self).__init__()
        self.branch_0 = Conv2d(in_channels, 384, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False),
            Conv2d(192, 224, (1, 7), stride=1, padding=(0, 3), bias=False),
            Conv2d(224, 256, (7, 1), stride=1, padding=(3, 0), bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False),
            Conv2d(192, 192, (7, 1), stride=1, padding=(3, 0), bias=False),
            Conv2d(192, 224, (1, 7), stride=1, padding=(0, 3), bias=False),
            Conv2d(224, 224, (7, 1), stride=1, padding=(3, 0), bias=False),
            Conv2d(224, 256, (1, 7), stride=1, padding=(0, 3), bias=False)
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            Conv2d(in_channels, 128, 1, stride=1, padding=0, bias=False)
        )
    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class Reduction_B(nn.Module):
    # 17 -> 8
    def __init__(self, in_channels):
        super(Reduction_B, self).__init__()
        self.branch_0 = nn.Sequential(
            Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False),
            Conv2d(192, 192, 3, stride=2, padding=0, bias=False),
        )
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 256, 1, stride=1, padding=0, bias=False),
            Conv2d(256, 256, (1, 7), stride=1, padding=(0, 3), bias=False),
            Conv2d(256, 320, (7, 1), stride=1, padding=(3, 0), bias=False),
            Conv2d(320, 320, 3, stride=2, padding=0, bias=False)
        )
        self.branch_2 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        return torch.cat((x0, x1, x2), dim=1)  # 8 x 8 x 1536


class Inception_C(nn.Module):
    def __init__(self, in_channels):
        super(Inception_C, self).__init__()
        self.branch_0 = Conv2d(in_channels, 256, 1, stride=1, padding=0, bias=False)

        self.branch_1 = Conv2d(in_channels, 384, 1, stride=1, padding=0, bias=False)
        self.branch_1_1 = Conv2d(384, 256, (1, 3), stride=1, padding=(0, 1), bias=False)
        self.branch_1_2 = Conv2d(384, 256, (3, 1), stride=1, padding=(1, 0), bias=False)

        self.branch_2 = nn.Sequential(
            Conv2d(in_channels, 384, 1, stride=1, padding=0, bias=False),
            Conv2d(384, 448, (3, 1), stride=1, padding=(1, 0), bias=False),
            Conv2d(448, 512, (1, 3), stride=1, padding=(0, 1), bias=False),
        )
        self.branch_2_1 = Conv2d(512, 256, (1, 3), stride=1, padding=(0, 1), bias=False)
        self.branch_2_2 = Conv2d(512, 256, (3, 1), stride=1, padding=(1, 0), bias=False)

        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            Conv2d(in_channels, 256, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x1_1 = self.branch_1_1(x1)
        x1_2 = self.branch_1_2(x1)
        x1 = torch.cat((x1_1, x1_2), 1)
        x2 = self.branch_2(x)
        x2_1 = self.branch_2_1(x2)
        x2_2 = self.branch_2_2(x2)
        x2 = torch.cat((x2_1, x2_2), dim=1)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1) # 8 x 8 x 1536


class Inceptionv4(nn.Module):
    def __init__(self, in_channels=3, classes=1000, k=192, l=224, m=256, n=384):
        super(Inceptionv4, self).__init__()
        blocks = []
        blocks.append(Stem(in_channels))
        for i in range(4):
            blocks.append(Inception_A(384))
        blocks.append(Reduction_A(384, k, l, m, n))
        for i in range(7):
            blocks.append(Inception_B(1024))
        blocks.append(Reduction_B(1024))
        for i in range(3):
            blocks.append(Inception_C(1536))
        self.features = nn.Sequential(*blocks)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1536, classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_average_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class RLNetPlaceHolder(nn.Module):

    def __init__(self, action_size=2, size_mem=256, aditional_size=12):
        super(RLNetPlaceHolder, self).__init__()
        #self.encoder = Inceptionv4()

        self.aditional_size = aditional_size

        self.hidden = nn.GRUCell(1000 + aditional_size + 32, size_mem)

        self.msg_layer = nn.Linear(size_mem, 32)

        self.mu = nn.Linear(size_mem, action_size)

        self.log_std = nn.Linear(size_mem, action_size)

    def forward(self, x, msg):

        batch_size = x[0].shape[0]
        dev = x[0].device
        x, aditional, hidden = torch.randn((batch_size, 1000)).to(dev), torch.randn((batch_size, 12)).to(dev), torch.randn((batch_size, 256)).to(dev)
        #x = self.encoder(x)

        aditional_flat = aditional.reshape(-1, self.aditional_size)
        
        x = x.reshape(-1, 1000)

        x_aug = torch.cat((x, aditional_flat), dim=1)

        hidden = hidden.detach()

        hidden = self.hidden(x_aug, hidden)

        mu = self.mu(F.relu(hidden))

        log_std = self.log_std(F.relu(hidden))


        return mu, log_std, hidden, self.msg_layer(F.relu(hidden))




class RLNetCriticPlaceHolder(nn.Module):

    def __init__(self, action_size=2, size_mem=256, aditional_size=12):
        super(RLNetCriticPlaceHolder, self).__init__()
        #self.encoder = Inceptionv4()

        self.aditional_size = aditional_size

        self.hidden = nn.GRUCell(1000 + aditional_size + 2 + 32, size_mem)

        self.msg_layer = nn.Linear(size_mem, 32)

        self.critic = nn.Linear(size_mem, 1)

    def forward(self, x, action, msg):

        batch_size = x[0].shape[0]
        dev = x[0].device

        x, aditional, hidden, action_ = torch.randn((batch_size, 1000)).to(dev), torch.randn((batch_size, 12)).to(dev), torch.randn((batch_size, 256)).to(dev), torch.randn((batch_size, 2)).to(dev)

        aditional_flat = aditional.reshape(-1, self.aditional_size)
        
        x = x.reshape(-1, 1000)


        #import pudb; pudb.set_trace()
        x_aug = torch.cat((x, aditional_flat, action_, msg), dim=1)

        hidden = hidden.detach()

        hidden = self.hidden(x_aug, hidden)

        return self.critic(F.relu(hidden)), self.msg_layer(F.relu(hidden))

        


class RLNet(nn.Module):

    def __init__(self, action_size=2, size_mem=256, aditional_size=12):
        super(RLNet, self).__init__()
        self.encoder = Inceptionv4()

        self.aditional_size = aditional_size

        self.hidden = nn.GRUCell(1000 + aditional_size, size_mem)

        self.mu = nn.Linear(size_mem, action_size)

        self.log_std = nn.Linear(size_mem, action_size)

    def forward(self, all_x):

        x, aditional, hidden = all_x
        x = self.encoder(x)

        aditional_flat = aditional.reshape(-1, self.aditional_size)
        
        x = x.reshape(-1, 1000)

        x_aug = torch.cat((x, aditional_flat), dim=1)

        hidden = hidden.detach()

        hidden = self.hidden(x_aug, hidden)

        mu = self.mu(F.relu(hidden))

        log_std = self.log_std(F.relu(hidden))


        return mu, log_std, hidden

        

class RLNetSum(nn.Module):

    def __init__(self, action_size=2, size_mem=256, aditional_size=12):
        super(RLNetSum, self).__init__()
        self.encoder = Inceptionv4()

        self.aditional_size = aditional_size

        self.hidden = nn.GRUCell(1000 + aditional_size, size_mem)

        self.mu = nn.Linear(size_mem, action_size)

        self.log_std = nn.Linear(size_mem, action_size)


    def forward(self, all_x):
        x, aditional, hidden = (torch.randn((4, 3, 300, 900)).cuda(), torch.randn((4, 2, 6)).cuda(), torch.randn((4, 256)).cuda()) 
        x = self.encoder(x)

        aditional_flat = aditional.reshape(-1, self.aditional_size)
        
        x = x.reshape(-1, 1000)

        x_aug = torch.cat((x, aditional_flat), dim=1)

        hidden = hidden.detach()

        hidden = self.hidden(x_aug, hidden)

        mu = self.mu(F.relu(hidden))

        log_std = self.log_std(F.relu(hidden))


        return mu, log_std, hidden



class RLNetCritic(nn.Module):

    def __init__(self, action_size=2, size_mem=256, aditional_size=12):
        super(RLNetCritic, self).__init__()
        self.encoder = Inceptionv4()

        self.aditional_size = aditional_size

        self.hidden = nn.GRUCell(1000 + aditional_size + 2, size_mem)

        self.critic = nn.Linear(size_mem, 1)



    def forward(self, all_x, action):
        x, aditional, hidden = all_x

        x = self.encoder(x)

        aditional_flat = aditional.reshape(-1, self.aditional_size)
        
        x = x.reshape(-1, 1000)


        #import pudb; pudb.set_trace()
        x_aug = torch.cat((x, aditional_flat, action), dim=1)

        hidden = hidden.detach()

        hidden = self.hidden(x_aug, hidden)

        return self.critic(F.relu(hidden))


effnet_model = EfficientNet.from_pretrained('efficientnet-b1')
tfms = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

shared_encoder = EfficientNet.from_pretrained('efficientnet-b0')

class RLNetMSG(nn.Module):

    def __init__(self, action_size=2, size_mem=256, aditional_size=12, msg_size=32):
        super(RLNetMSG, self).__init__()
        self.encoder = shared_encoder

        self.aditional_size = aditional_size

        self.hidden = nn.GRUCell(1000 + aditional_size + msg_size, size_mem)

        self.hidden_critic = nn.GRUCell(1000 + aditional_size + msg_size + 2, size_mem)

        self.critic_layer = nn.Linear(size_mem, 1)

        self.msg_layer = nn.Linear(size_mem, msg_size)

        self.mu = nn.Linear(size_mem, action_size)

        self.log_std = nn.Linear(size_mem, action_size)


    def forward(self, all_x, msg):

        x, aditional, hidden = all_x

        x = self.encoder(x)

        aditional_flat = aditional.reshape(-1, self.aditional_size)
        
        x = x.reshape(-1, 1000)        

        x_aug = torch.cat((x, aditional_flat, msg), dim=1)

        hidden = hidden.detach()

        hidden = self.hidden(x_aug, hidden)

        mu = self.mu(F.relu(hidden))

        log_std = self.log_std(F.relu(hidden))


        return mu, log_std, hidden, self.msg_layer(F.relu(hidden))


    def critic(self, all_x, action, msg):

        x, aditional, hidden = all_x

        x = self.encoder(x)

        aditional_flat = aditional.reshape(-1, self.aditional_size)
        
        x = x.reshape(-1, 1000)

        x_aug = torch.cat((x, aditional_flat, action, msg), dim=1)

        hidden = hidden.detach()

        hidden = self.hidden_critic(x_aug, hidden)

        return self.critic_layer(F.relu(hidden)), self.msg_layer(F.relu(hidden))


class RLNetCriticMSG(nn.Module):

    def __init__(self, action_size=2, size_mem=256, aditional_size=12, msg_size=32):
        super(RLNetCriticMSG, self).__init__()
        self.encoder = shared_encoder

        self.aditional_size = aditional_size

        self.hidden = nn.GRUCell(1000 + aditional_size + 2 + msg_size, size_mem)

        self.msg_layer = nn.Linear(size_mem, msg_size)

        self.critic = nn.Linear(size_mem, 1)



    def forward(self, all_x, action, msg):
        x, aditional, hidden = all_x

        #x = tfms(x)
        x = self.encoder(x)

        aditional_flat = aditional.reshape(-1, self.aditional_size)
        
        x = x.reshape(-1, 1000)

        
        #batch_size = aditional_flat.shape[0]
        #dev = x[0].device

        #x = torch.randn((batch_size, 1000)).to(dev)

        x_aug = torch.cat((x, aditional_flat, action, msg), dim=1)

        hidden = hidden.detach()

        hidden = self.hidden(x_aug, hidden)

        return self.critic(F.relu(hidden)), self.msg_layer(F.relu(hidden))









if __name__ == '__main__':

    from torchsummary import summary
    
    #image_im = torch.randn((4, 3, 300, 900))
    
    obs_imgs = torch.randn((4, 3, 300, 900))
    hiddens = torch.randn((4, 256))
    aditionals = torch.randn((4, 2, 6))
    action = torch.randn((4, 1, 2))

    #rl_model = RLNetSum().cuda()

    #place_holder_pol = RLNetPlaceHolder()
    #place_holder_critic = RLNetCriticPlaceHolder()

    #place_holder_pol(obs_imgs)
    #place_holder_critic(place_holder_pol, action)

    #in_channels, action_size, aditional_size
    #rl_model = ResNetRLGRU(3, 2, 12)

    #model = resnet101(3, 4)
    #res = rl_model('ergr')
    #print(res)

    #float_lstm = lstm_for_demonstration(model_dimension, model_dimension,lstm_depth)

    # this is the call that does the work
    #quantized_lstm = torch.quantization.quantize_dynamic(
    #    float_lstm, {nn.LSTM, nn.Linear}, dtype=torch.qint8
    #)

    #quantized_model = torch.quantization.quantize_dynamic(
    #    rl_model, {nn.GRUCell, nn.Linear}, dtype=torch.qint8
    #    ).cuda()


    # specify quantization config for QAT
    #qat_model.qconfig=torch.quantization.get_default_qat_qconfig('fbgemm')

    # prepare QAT
    #torch.quantization.prepare_qat(qat_model, inplace=True)

    # convert to quantized version, removing dropout, to check for accuracy on each
    #epochquantized_model=torch.quantization.convert(qat_model.eval(), inplace=False)



    #summary(rl_model, (3, 300, 900))
