import os
import torch
import glob
import numpy 
import cv2


import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets


img_dir = '/home/gonvas/Programming/carla_scenario/CarlaDRL/vae_final/'

#import pudb; pudb.set_trace()

images_folder = datasets.ImageFolder(root=img_dir)

#dataset_loader = torch.utils.data.DataLoader(images_folder,
#                                             batch_size=4, shuffle=True,
#                                             num_workers=4)

counter = 0

for img, _ in images_folder:
	img_tensor = transforms.ToTensor()(img).unsqueeze_(0)
	rgb_front = img_tensor[:, :, :, 0:300]
	rgb_back = img_tensor[:, :, :, 300:600]   
	minimap = img_tensor[:, :, :, 600:900]   

	transforms.ToPILImage()(rgb_front.squeeze_(0)).save(img_dir+'img_data/'+'rgbfront/env_front_0_{}.png'.format(counter))
	transforms.ToPILImage()(rgb_back.squeeze_(0)).save(img_dir+'img_data/'+'rgbback/env_back_0_{}.png'.format(counter))
	transforms.ToPILImage()(minimap.squeeze_(0)).save(img_dir+'img_data/'+'minimap/env_minimap_0_{}.png'.format(counter))


	if(counter != 0 and counter%100 == 0):
		print('Counter : {:4d}, Total Amount: {:4d}'.format(counter, len(images_folder)))	

	counter += 1