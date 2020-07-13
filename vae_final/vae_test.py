
from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
import torch.nn.functional as nnf
import time
from glob import glob
import numpy as np
from PIL import Image
import sys
from math import ceil, pow
import math
from random import choice


import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import wandb

import cv2

class EnvImagesDataset(Dataset):
    def __init__(self, filelist, batch_size, maxram, device, size_per_image=300*900*3*4):

        self.device = device
        self.filelist = filelist
        print('Amount of images of carla env : {}'.format(len(filelist)))

        size_img_batch = size_per_image*batch_size
        n_images = math.floor(maxram*1_000_000_000 / size_per_image)

        # Number of images needs to be pair
        n_images = math.ceil(n_images / 2.) * 2

        n_images = min(len(filelist) - 2, n_images)


        #self.all_images = [(torch.FloatTensor(cv2.imread(img_path))/255).transpose(0,2).transpose(1,2) for img_path in filelist]
        #self.all_images = [(transforms.ToTensor()(Image.open(im_loc))) for im_loc in filelist[:n_images]]

        self.max_len = n_images

        #import pudb; pudb.set_trace()

        #final_img_bytes = int(len(self.all_images)*self.all_images[0].nelement()*self.all_images[0].element_size())
        
        #print('Dataset: Amount of images {}, Total Size {} in Mbytes, Image Size {} Mbytes '.format(len(self.all_images), final_img_bytes/1_000_000, size_per_image/1_000_000))
        #print('Batch Size: {}'.format(batch_size))

    def __len__(self):
        return 20 * round((len(self.filelist) - 20)/20)
        #return len(self.all_images)


    def __getitem__(self, idx):

        #image = (transforms.ToTensor()(Image.open(im_loc))) for im_loc in filelist[:n_images]
        try:
            image = (transforms.ToTensor()(Image.open(self.filelist[idx])))
        except:
            image = torch.FloatTensor(torch.randn(3, 300, 900))
        #image = self.all_images[idx]

        return image.to(self.device)




def conv_output_shape(h_w, kernel_size=3, stride=1, pad=0, dilation=1):
    
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


class VAE(nn.Module):
    def __init__(self, state_size=((256, 256), 3), conv_cahannels=8,  latent_variable_size=256, hidden_dims = None, noise=False):
        super(VAE, self).__init__()

        self.noise = noise
        
        self.state_size, in_channels = state_size

        self.ngf = conv_cahannels
        self.ndf = conv_cahannels

        ngf = self.ngf
        ndf = self.ndf

        self.latent_variable_size = latent_variable_size
        latent_dim = latent_variable_size


        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        

        self.fc_mu = nn.Linear(32768, latent_dim)
        self.fc_var = nn.Linear(32768, latent_dim)
        #self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        #self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())



    def add_noise(self, image):

        std = 0.2
        mean = 0

        def gaussian(image):
            image = image + torch.randn(image.size()).to(image.device) * std + mean
            return image

        def cut_img(image):
            if torch.rand(1).item() < .5:
                image[:, :, :int(torch.rand(1).item()*0.85*int(self.state_size[0]/2)), :] = 0
            else:
                image[:, :, int(torch.rand(1).item()*int(self.state_size[0])) + int(self.state_size[0]/2) - 1:, :] = 0
            return image

        def color_noise(image):
            if torch.rand(1).item() < .5:
                image = image * 0.5
            else:
                image = image * 2
            return image

        noise_lists = [gaussian, color_noise, cut_img]

        noise_to_apply = choice(noise_lists)
        return noise_to_apply(image)
        


    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]


    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)

        result = self.final_layer(result)

        return nnf.interpolate(result, size=(256, 256), mode='bicubic', align_corners=False)


    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


    def forward(self, inputs):
        #import pudb; pudb.set_trace()
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        final_recons = self.decode(z)
        #print('Mu max, min: {}, {}, Log_var: {}, {}'.format(mu.max(), mu.min(), log_var.max(), log_var.min()))
        return  [final_recons, inputs, mu, log_var]


    def loss_function(self, recons, input_, mu, log_var, bs_total_size):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """


        kld_weight = 0.005 # batchsize/total_size
        
        #import pudb; pudb.set_trace()
        recons_loss = F.mse_loss(recons, input_)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return loss
        #return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}


    def rand_samples(self, device, amount=8):
        mus = torch.rand((amount, self.latent_variable_size)) * 6 - 3 
        log_vars = torch.rand((amount, self.latent_variable_size)) * (1.7 * 2) - 1.7
        z = self.reparameterize(mus.to(device), log_vars.to(device))

        final_recons = self.decode(z)
        
        #print('Mu max, min: {}, {}, Log_var: {}, {}'.format(mu.max(), mu.min(), log_var.max(), log_var.min()))
        return  [final_recons, None, mus, log_vars]



    #def latent_space_imagination(self, device):   
    #    import pudb; pudb.set_trace()


    #def lin_space_latent(self):
    #    mus = 

    #def latent_space
    """ 
    def encode(self, x):
        
        if(self.noise):
            x = self.add_noise(x)


        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        #h5 = h5.view(-1, self.ndf*8*4*4)

        h5 = h5.reshape(-1, self.flat_size)

        return self.fc1(h5), self.fc2(h5)

    def reparametrize(self, mu, logvar):
        #std = logvar.mul(0.5).exp_()

        std = torch.exp(0.5 * logvar)

        #if args.cuda:
        #    eps = torch.cuda.FloatTensor(std.size()).normal_()
        #else:
        #    eps = torch.FloatTensor(std.size()).normal_()
        #eps = torch.cuda.FloatTensor(std.size()).normal_()
        #eps = Variable(eps)
        #eps = torch.normal(0, 1, size=(mu.size(), ))
        eps = torch.normal(0, 1, size=(mu.shape[0], 1)).to(std.device)

        return mu + std * eps


    def decode(self, z):
        h1 = self.relu(self.d1(z))
        
        h1 = h1.reshape(self.unflatt_size)
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
        h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))
        h6 = self.sigmoid(self.d6(self.pd5(self.up5(h5))))

        diff_h = self.state_size[0] - h6.shape[2]
        diff_w = self.state_size[1] - h6.shape[3]

        h7 = F.pad(h6, (0, diff_w, 0, diff_h), mode='replicate')

        return h7

    def get_latent_var(self, x):
        mu, logvar = self.encode(x.reshape(-1, self.nc, self.ndf, self.ngf))
        #mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x):
        #mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        #import pudb; pudb.set_trace()
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar


    def loss(self, input_x, recons, mu, log_var):

        recons_loss = F.mse_loss(recons, input_x)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + input_x.shape[0] * kld_loss

        return loss + recons_loss
    """

"""
reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False
def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)

    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)


    return BCE + KLD
"""
"""
    train_loader = torch.utils.data.DataLoader(
                            torchvision.datasets.MNIST('./data/mnist', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
            batch_size=512, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
                            torchvision.datasets.MNIST('./data/mnist', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
            batch_size=64, shuffle=True)
"""



def train_vae(device, images_loc, saves_loc, epochs, log_step=50, test_epoch=10):

    images_dataset = torchvision.datasets.ImageFolder('/home/gonvas/Programming/carla_scenario/CarlaDRL/vae_final/img_data/rgbfront', transform=transforms.Compose([
        torchvision.transforms.Resize((256, 256), interpolation=2),
        transforms.ToTensor()]))


    #import pudb; pudb.set_trace()

    lengths = [int(len(images_dataset)*0.9), int(len(images_dataset)*0.1)]

    remaing_images = len(images_dataset) - (lengths[0] + lengths[1])
    lengths[0] += remaing_images

    train_set, val_set = torch.utils.data.random_split(images_dataset, lengths)

    train_loader = torch.utils.data.DataLoader(train_set,
                         batch_size=64,
                         num_workers=4,
                         shuffle=True)

    test_loader = torch.utils.data.DataLoader(val_set,
                         batch_size=32,
                         num_workers=4,
                         shuffle=True)
    
    test_iter = iter(test_loader)

    vae_model = VAE().to(device)

    optimizer = optim.Adam(vae_model.parameters(), lr=1e-4)

    hypeparameters = {'latent_size':256, 'image_size':256, 'kld_weight':0.005}

    wandb.init(config=hypeparameters)
    wandb.watch(vae_model)


    wandb.log({'images_numb':len(train_loader), 'batch_size': 128})

    #loader = DataLoader(dataset, batch_size=10, num_workers=5)
    #mean: tensor(0.0001)
    #std: tensor(0.2933)

    """
    mean = 0.0
    meansq = 0.0
    count = 0

    for index, data in enumerate(train_loader):
        mean = data[0].sum()
        meansq = meansq + (data[0]**2).sum()
        count += np.prod(data[0].shape)


    total_mean = mean/count
    total_var = (meansq/count) - (total_mean**2)
    total_std = torch.sqrt(total_var)
    print("mean: " + str(total_mean))
    print("std: " + str(total_std))
    """

    last_loss = 0

    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            #data = load_batch(batch_idx, True)
            #import pudb; pudb.set_trace()
            data = data[0].to(device)

            optimizer.zero_grad()

            #import pudb; pudb.set_trace()
            recon_batch, input_, mu, logvar = vae_model(data)

            #import pudb; pudb.set_trace()

            #[self.decode(z), input, mu, log_var]

            #loss = vae_model.loss(data, recon_batch, mu, logvar)

            loss_weight = len(data)/(len(train_loader)*len(data))
            #recons, intputs, mu, log_var, bs_total_size
            loss = vae_model.loss_function(recon_batch, data, mu, logvar, loss_weight)
            loss.backward()
            
            train_loss += loss.item()
            optimizer.step()

            #if(batch_idx != 0 and batch_idx % (len(train_loader)/log_mul) == 0):
            if(batch_idx != 0 and batch_idx % log_step == 0):
                print('Epoch: {}, Train Loss: {}, this batch_loss : {}'.format(epoch, train_loss, loss.item()))

            wandb.log({'epoch':epoch, 'train_loss': train_loss, 'batch_loss':loss.item()})

        last_loss = train_loss

        if epoch != 0 and epoch % test_epoch == 0:
            img_test = test_iter.next()[0]

            recons, _, _, _ = vae_model(img_test.to(device))



            wandb.log({'input_image': wandb.Image(torchvision.utils.make_grid(img_test.cpu()))})
            wandb.log({'recons_image': wandb.Image(torchvision.utils.make_grid(recons.cpu()))})

            #torchvision.utils.save_image(recons, './test_images/test_{}_recon.jpg'.format(epoch))
            #torchvision.utils.save_image(img_test, './test_images/test_{}_input.jpg'.format(epoch))
            print('Saved Images of testing')

        if epoch != 0 and epoch % (test_epoch*10) == 0:

            rand_samples, _ , _, _ = vae_model.rand_samples(device)
            wandb.log({'rand_sample': wandb.Image(torchvision.utils.make_grid(rand_samples.cpu()))})


            print('Sampled Latent Space')

    print('Finished training, saving')

    save_dir = './'

    torch.save({
        'epochs': epochs,
        'model_state_dict': vae_model.state_dict(),
        'loss': last_loss,
        'latent_vars': 256,
        }, save_dir+'final_vae_model.tar')

    wandb.save(save_dir+'final_vae_model.tar')


train_vae(torch.device('cuda'), None, None, 100)