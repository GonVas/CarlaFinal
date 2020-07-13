from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split
import cv2
from math import ceil
import glob


import numpy as np

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=90, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
"""
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
"""


imgh = int(720*0.5)
imgw = int(1280*0.5)
img_in_channels = 3
latent_size = 300
conv_channels = 8
kernel_size = 2

len_imgs_batch_mul = 32
sampling_size = 128

filelist = glob.glob('/run/media/gonvas/toshiba-ssd/football/images/test3/*.png')[:args.batch_size*len_imgs_batch_mul]

save_times = 50




class EnvImagesDataset(Dataset):
    def __init__(self, data_root=filelist):

        self.all_images = [(torch.FloatTensor(cv2.imread(img_path))/255).transpose(0,2).transpose(1,2) for img_path in filelist]


    def __len__(self):  
        return len(self.all_images)

    def __getitem__(self, idx):
        return self.all_images[idx]



dataset = EnvImagesDataset()
split_ratio = 0.75
train_size, test_size = int(len(dataset)*split_ratio), int(len(dataset)*(1-split_ratio))
trainset, valset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=3)
val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=3)

class VAE(nn.Module):
    def __init__(self, state_size=((imgh, imgw), img_in_channels), latent_size=latent_size, conv_channels=conv_channels, kernel_size=kernel_size):
        super(VAE, self).__init__()

        self.state_size, self.channels_in = state_size

        self.conv1 = nn.Conv2d(self.channels_in, conv_channels, kernel_size=kernel_size, stride=1, padding=0)
        
        self.bn1 = nn.BatchNorm2d(num_features=conv_channels)

        self.pool1 = nn.MaxPool2d((2, 2), return_indices=True)
        self.conv2 = nn.Conv2d(conv_channels, conv_channels*2, kernel_size=kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=conv_channels*2)
        self.pool2 = nn.MaxPool2d((2, 2), return_indices=True)
        self.conv3 = nn.Conv2d(conv_channels*2, conv_channels*2, kernel_size=kernel_size, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=conv_channels*2)
        self.pool3 = nn.MaxPool2d((2, 2), return_indices=True)


        self.flat_size = (-1, conv_channels*2 * ceil(self.state_size[0]/8) * ceil(self.state_size[1]/8))


        self.fc21 = nn.Linear(conv_channels*2 * ceil(self.state_size[0]/8) * ceil(self.state_size[1]/8), latent_size)
        self.fc22 = nn.Linear(conv_channels*2 * ceil(self.state_size[0]/8) * ceil(self.state_size[1]/8), latent_size)

        self.fc3 = nn.Linear(latent_size, conv_channels*2 * ceil(self.state_size[0]/8) * ceil(self.state_size[1]/8))

        self.unflatt_size = (-1, conv_channels*2, ceil(self.state_size[0]/8), ceil(self.state_size[1]/8))

        self.unpool1 = nn.MaxUnpool2d(kernel_size)
        self.uncov1 = nn.Conv2d(conv_channels*2, conv_channels*2, kernel_size=kernel_size, padding=0)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.unpool2 = nn.MaxUnpool2d(kernel_size)
        self.uncov2 = nn.Conv2d(conv_channels*2, conv_channels, kernel_size=kernel_size, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.unpool3 = nn.MaxUnpool2d(kernel_size)
        self.uncov3 = nn.Conv2d(conv_channels, self.channels_in, kernel_size=kernel_size, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')


    def encode(self, x):    
        
        x = self.bn1(self.conv1(x))
        x, self.indices1 = self.pool1(x)
        x = F.relu(x)
        self.size_for2_x = x.size()

        x = self.bn2(self.conv2(x))
        x, self.indices2 = self.pool2(x)
        x = F.relu(x)
        self.size_for3_x = x.size()


        x = self.bn3(self.conv3(x))
        x, self.indices3 = self.pool3(x)
        x = F.relu(x)


        h1 = x.reshape(self.flat_size)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, only_decode=False):

        if(only_decode):

            if(len(z) > len(self.indices1)):
                self.indices1 = self.indices1.repeat(int(len(z)/len(self.indices1)), 1 , 1, 1)
                self.indices2 = self.indices2.repeat(int(len(z)/len(self.indices2)), 1 , 1, 1)
                self.indices3 = self.indices3.repeat(int(len(z)/len(self.indices3)), 1 , 1, 1)

            #import pudb; pudb.set_trace()

        
        h3 = self.fc3(z)
        h3 = h3.view(self.unflatt_size)

        #h3 = self.upsample1(h3)
        h3 = self.unpool1(h3, self.indices3)
        h3 = F.elu(self.uncov1(h3))

        h3 = F.pad(h3, (0,1,0,1), mode='replicate')

        #h3 = self.upsample2(h3)
        h3 = self.unpool2(h3, self.indices2)
        h3 = F.elu(self.uncov2(h3))

        h3 = F.pad(h3, (0,-2,0,-2), mode='replicate')
        h3 = self.unpool3(h3, self.indices1)
        h3 = F.elu(self.uncov3(h3))
        #h3 = self.upsample3(h3)

        h3 = F.pad(h3, (0,1,0,1), mode='replicate')

        return torch.sigmoid(h3)

    def sample(self, amount_multiplier):
        sample = torch.randn(amount_multiplier*args.batch_size, latent_size).to(device)
        sample = model.decode(sample, only_decode=True).cpu()
        return sample


    def special_sample(self, amount_multiplier):
        sample = torch.randn(amount_multiplier*args.batch_size, latent_size).to(device)
        sample = model.decode(sample, only_decode=True).cpu()
        return sample


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def conv_output_shape(self, h_w, kernel_size=3, stride=1, pad=0, dilation=1):
        """
        Utility function for computing output of convolutions
        takes a tuple of (h,w) and returns a tuple of (h,w)
        """
        
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

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


"""
def get_data_batches():
    #batches_data = []
   
    for i in range(0, len(filelist), args.batch_size):
        #import pudb; pudb.set_trace()
        print('Grabbing a batch i : {}'.format(i))
        yield torch.IntTensor([cv2.imread(img_path) for img_path in filelist[i:i + args.batch_size]]).transpose(1,3).transpose(2,3).to(device) 
"""

def train(epoch):

    model.train()
    train_loss = 0
    print('Batches per epoch {}'.format(len(train_loader)))
    #import pudb; pudb.set_trace()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0 and batch_idx != 0 :
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
        
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / (len(train_loader))))



def test(epoch):

    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 4)
                comparison = torch.cat([data[:n], recon_batch.view(val_loader.batch_size, img_in_channels, imgh, imgw)[:n]])
                save_image(comparison.cpu(), 'results/reconstruction_big_' + str(epoch) + '.jpeg', nrow=n)
    
    test_loss /= len(val_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    """
    if(epoch == 2):
        n = 15  # figure with 15x15 digits
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        # we will sample n points within [-15, 15] standard deviations
        grid_x = np.linspace(-15, 15, n)
        grid_y = np.linspace(-15, 15, n)
        import pudb; pudb.set_trace()
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = torhc.Tensor(np.array([[xi, yi]]) * epsilon_std)
                x_decoded = generator.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure)
        plt.show()
    """


if __name__ == "__main__":


    for epoch in range(1, args.epochs + 1):
        train(epoch)

        if((epoch + 1) % save_times == 0):
            test(epoch)
            with torch.no_grad():
                #sample = torch.randn(128, latent_size).to(device)
                sample = model.sample(4).cpu()
                save_image(sample.view(4*args.batch_size, 3, imgh, imgw),
                           'results/sample_big_' + str(epoch) + '.jpeg')