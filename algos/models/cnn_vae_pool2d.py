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
import time
from glob import glob
from .util import *
import numpy as np
from PIL import Image
import sys
from math import ceil, pow
import math
from random import choice


import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/vae/')



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


"""
dataset = EnvImagesDataset()
split_ratio = 0.75
train_size, test_size = int(len(dataset)*split_ratio), int(len(dataset)*(1-split_ratio))
trainset, valset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)#, num_workers=3)
test_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=True)#, num_workers=3)
"""




class VAE(nn.Module):
    def __init__(self, state_size=((28, 28), 1), conv_cahannels=8,  latent_variable_size=400, noise=True):
        super(VAE, self).__init__()

        self.noise = noise
        
        self.state_size, self.channels_in = state_size

        self.ngf = conv_cahannels
        self.ndf = conv_cahannels

        ngf = self.ngf
        ndf = self.ndf

        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv2d(self.channels_in, ndf, 2, 2, 1)
        self.bn1 = nn.BatchNorm2d(ndf)

        #self.size_now = conv_output_shape(self, h_w, kernel_size=3, stride=1, pad=0, dilation=1)
        self.size_now = conv_output_shape(self.state_size, kernel_size=2, stride=2, pad=1)

        self.e2 = nn.Conv2d(ndf, ndf*2, 2, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        self.size_now = conv_output_shape(self.size_now, kernel_size=2, stride=2, pad=1)

        self.e3 = nn.Conv2d(ndf*2, ndf*4, 2, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        self.size_now = conv_output_shape(self.size_now, kernel_size=2, stride=2, pad=1)

        self.e4 = nn.Conv2d(ndf*4, ndf*8, 2, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf*8)

        self.size_now = conv_output_shape(self.size_now, kernel_size=2, stride=2, pad=1)

        self.e5 = nn.Conv2d(ndf*8, ndf*8, 2, 2, 1)
        self.bn5 = nn.BatchNorm2d(ndf*8)


        self.size_now = conv_output_shape(self.size_now, kernel_size=2, stride=2, pad=1)
        
        self.flat_size = int(self.size_now[0])*int(self.size_now[1])*ndf*8

        self.fc1 = nn.Linear(self.flat_size, self.latent_variable_size)
        self.fc2 = nn.Linear(self.flat_size, self.latent_variable_size)


        self.unflatt_size = (-1, ndf*8, int(self.size_now[0]), int(self.size_now[1]))

        # decoder
        self.d1 = nn.Linear(self.latent_variable_size, self.flat_size)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(ngf*8, ngf*8, 3, 1)
        self.bn6 = nn.BatchNorm2d(ngf*8, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(ngf*8, ngf*4, 3, 1)
        self.bn7 = nn.BatchNorm2d(ngf*4, 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(ngf*4, ngf*2, 3, 1)
        self.bn8 = nn.BatchNorm2d(ngf*2, 1.e-3)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(1)
        self.d5 = nn.Conv2d(ngf*2, ngf, 3, 1)
        self.bn9 = nn.BatchNorm2d(ngf, 1.e-3)

        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6 = nn.Conv2d(ngf, self.channels_in, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


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
        std = logvar.mul(0.5).exp_()
        #if args.cuda:
        #    eps = torch.cuda.FloatTensor(std.size()).normal_()
        #else:
        #    eps = torch.FloatTensor(std.size()).normal_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

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




#if args.cuda:
#    model.cuda()

reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False
def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)

    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)


    return BCE + KLD


def train(model, train_loader, epoch, cuda=True, log_interval=100):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        #data = load_batch(batch_idx, True)
        #import pudb; pudb.set_trace()
        data = Variable(data)
        if cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), (len(train_loader)*128),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / (len(train_loader)*128)))
    return train_loss / (len(train_loader)*128)


def test(model, test_loader, epoch, cuda=True):
    model.eval()
    test_loss = 0
    for batch_idx, data in enumerate(test_loader):
        #data = load_batch(batch_idx, False)
        #import pudb; pudb.set_trace()
        data = Variable(data, volatile=True)
        if cuda:
            data = data.cuda()
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).item()

        if(batch_idx == 0):
            torchvision.utils.save_image(data.data, './data/recons/Epoch_{}_data.jpg'.format(epoch), nrow=8, padding=2)
            #img_data = Image.open('./recons/Epoch_{}_data.jpg'.format(epoch))
        
            #r, g, b = img_data.split()
            #final_data = Image.merge("RGB", (b, g, r))
            #final_data.save('./recons/Epoch_{}_data.jpg'.format(epoch))      
            torchvision.utils.save_image(recon_batch.data, './data/recons/Epoch_{}_recon.jpg'.format(epoch), nrow=8, padding=2)
        
            #img_data = Image.open('./recons/Epoch_{}_recon.jpg'.format(epoch))
            #r, g, b = img_data.split()
            #final_data = Image.merge("RGB", (b, g, r))
            #final_data.save('./recons/Epoch_{}_recon.jpg'.format(epoch))  

            #writer.add_graph(model, data)



    test_loss /= (len(test_loader)*128)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


def perform_latent_space_arithmatics(items): # input is list of tuples of 3 [(a1,b1,c1), (a2,b2,c2)]
    load_last_model()
    model.eval()
    data = [im for item in items for im in item]
    data = [totensor(i) for i in data]
    data = torch.stack(data, dim=0)
    data = Variable(data, volatile=True)
    if args.cuda:
        data = data.cuda()
    z = model.get_latent_var(data.view(-1, model.nc, model.ndf, model.ngf))
    it = iter(z.split(1))
    z = zip(it, it, it)
    zs = []
    numsample = 11
    for i,j,k in z:
        for factor in np.linspace(0,1,numsample):
            zs.append((i-j)*factor+k)
    z = torch.cat(zs, 0)
    recon = model.decode(z)

    it1 = iter(data.split(1))
    it2 = [iter(recon.split(1))]*numsample
    result = zip(it1, it1, it1, *it2)
    result = [im for item in result for im in item]

    result = torch.cat(result, 0)
    torchvision.utils.save_image(result.data, './data/imgs/vec_math.jpg', nrow=3+numsample, padding=2)


def latent_space_transition(items): # input is list of tuples of  (a,b)
    load_last_model()
    model.eval()
    data = [im for item in items for im in item[:-1]]
    data = [totensor(i) for i in data]
    data = torch.stack(data, dim=0)
    data = Variable(data, volatile=True)
    if args.cuda:
        data = data.cuda()
    z = model.get_latent_var(data.view(-1, model.nc, model.ndf, model.ngf))
    it = iter(z.split(1))
    z = zip(it, it)
    zs = []
    numsample = 11
    for i,j in z:
        for factor in np.linspace(0,1,numsample):
            zs.append(i+(j-i)*factor)
    z = torch.cat(zs, 0)
    recon = model.decode(z)

    it1 = iter(data.split(1))
    it2 = [iter(recon.split(1))]*numsample
    result = zip(it1, it1, *it2)
    result = [im for item in result for im in item]

    result = torch.cat(result, 0)
    torchvision.utils.save_image(result.data, './data/imgs/trans.jpg', nrow=2+numsample, padding=2)


def rand_faces(num=5):
    load_last_model()
    model.eval()
    z = torch.randn(num*num, model.latent_variable_size)
    z = Variable(z, volatile=True)
    if args.cuda:
        z = z.cuda()
    recon = model.decode(z)
    torchvision.utils.save_image(recon.data, './data/rand_faces.jpg', nrow=num, padding=2)


def load_last_model(imgh, imgw):
    #models = glob('./savedmodels/vae/*.pth')
    models = sorted(glob('./savedmodels/vae/final_vae_*.pth'))


    if(len(models) == 0):
        print('No Saved Models')
        return None, None, None

    #model_ids = [(int(f.split('_')[1]), f) for f in models]
    #start_epoch, last_cp = max(model_ids, key=lambda item:item[0])
    #print('Last checkpoint: ', last_cp)

    model_to_load = VAE(state_size=((imgh, imgw), 3))
    model_to_load.load_state_dict(torch.load(models[-1], map_location='cpu'))
    #print('Loaded a model, last epoch {}'.format(start_epoch))
    return 1, 1, model_to_load


def resume_training(model, train_loader, test_loader, device, test_every=10, save_every=10, amount_epochs=100):
    #start_epoch, _ = load_last_model()
    start_epoch = 0
    for epoch in range(start_epoch + 1, start_epoch + amount_epochs + 1):
        train_loss = train(model, train_loader, epoch, device)
        if(epoch != 0 and epoch % test_every == 0):
            print('testingh')
            test_loss = test(model, test_loader, epoch, device)
        if(epoch != 0 and epoch % save_every == 0):
            torch.save(model.state_dict(), './project/savedmodels/vae/Epoch_{}_Train_loss_{:.4f}_Test_loss_{:.4f}.pth'.format(epoch, train_loss, test_loss))


def last_model_to_cpu():
    _, _, last_model = load_last_model()
    return last_model
    #torch.save(model.state_dict(), './savedmodels/vae/cpu_'+last_cp.split('/')[-1])



def vae_train(device, batch_size, maxram, max_epochs, log_interval, imgs_dir='./data/imgs/', model=None):
    imgh = 300
    imgw = 900
    n_channels = 3
    filelist = []
    all_images_locs = imgs_dir + '*.png'
    filelist = sorted(glob(all_images_locs))[:-15]


    dataset = EnvImagesDataset(filelist, batch_size, maxram, device)
    split_ratio = 0.75
    train_size, test_size = int(len(dataset)*split_ratio), int(len(dataset)*(1-split_ratio))
    trainset, valset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)#, num_workers=3) Dont shuffle glob does it
    test_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)#, num_workers=3)

    print('Train Loader : {} x {}'.format(len(train_loader), batch_size))

    if(model == None):
        model = VAE(state_size=((imgh, imgw), n_channels))

    resume_training(model.to(device), train_loader, test_loader, device, test_every=log_interval, save_every=log_interval, amount_epochs=max_epochs)


    writer.close()
    return model




if __name__ == '__main__':

    resume_training()
    # last_model_to_cpu()
    # load_last_model()
    # rand_faces(10)
    # da = load_pickle(test_loader[0])
    # da = da[:120]
    # it = iter(da)
    # l = zip(it, it, it)
    # # latent_space_transition(l)
    # perform_latent_space_arithmatics(l)