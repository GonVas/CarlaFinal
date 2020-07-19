from __future__ import print_function
import argparse
import os

import numpy as np
import time
import requests

import torch
import torch.multiprocessing as mp
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.optim as optim

from gym import error, spaces, make

import CarlaGymEnv
import sac_simple_channel 
import rl_human
import model_tester
#from algos import sac, random_agent, DQN_carla, policy_grads, reinforce, sac_mem, sac_a3c_dist, a3c, data_gatherer
#from algos.models.cnn_vae import VAE, load_last_model, vae_train


class TorchImgCarlaGymEnv:

    def __init__(self, env):
        self._env = env
        self.action_space = self._env.action_space


    def reset(self):
        pre_state = self._env.reset()
        return torch.Tensor(pre_state[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).float().squeeze(0)


    def step(self, action):
        pre_state, reward, done, info = self._env.step(action)
        return torch.Tensor(pre_state).unsqueeze(0).transpose(1, 3).transpose(2, 3).float().squeeze(0), reward, done, info


    def render(self):
        return self._env.render()



class TorchGymEnv:

    def __init__(self, env):
        self._env = env
        self.action_space = self._env.action_space


    def reset(self):
        pre_state = self._env.reset()
        return torch.Tensor(pre_state)


    def step(self, action):
        pre_state, reward, done, info = self._env.step(int(action))
        return torch.Tensor(pre_state), reward, done, info


    def render(self):
        return self._env.render()



class VaeEnvNoAddedInfo:

    def __init__(self, env, vae):

        self.vae = vae
        #self.observation_space = spaces.Box(low=0, high=1, shape=(1, vae.latent_variable_size), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(vae.latent_variable_size, ), dtype=np.float32)
        self._env = env
        self.action_space = env.action_space


    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = torch.Tensor(obs).unsqueeze(0).transpose(1, 3).transpose(2, 3).float().to(device)
        mu, logvar = self.vae.encode(obs)
        z = self.vae.reparametrize(mu, logvar)
        return z.to(torch.device("cpu")).detach(), reward, done, info


    def reset(self):
        pre_state = self._env.reset()
       
        obs = torch.Tensor(pre_state[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).float().to(device)
        mu, logvar = self.vae.encode(obs)
        z = self.vae.reparametrize(mu, logvar)
        return z.to(torch.device("cpu")).detach()


    def render(self):
        print('Rendering')


class VaeEnvAugmented:

    def __init__(self, env, vae, rank=0):
        self.rank = rank
        self.vae = vae
        #self.observation_space = spaces.Box(low=0, high=1, shape=(1, vae.latent_variable_size), dtype=np.float32)
        self.additional_obs_size = env.additional_obs_space.shape[0] * env.additional_obs_space.shape[1]
        self.observation_space = spaces.Box(low=0, high=1, shape=(vae.latent_variable_size + self.additional_obs_size, ), dtype=np.float32)
        self._env = env
        self.action_space = env.action_space


    def step(self, action):
        pre_state, reward, done, info = self._env.step(action)
        obs = torch.Tensor(pre_state[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).float().to(device)
        self.last_image = pre_state[0]
        mu, logvar = self.vae.encode(obs)
        z = self.vae.reparametrize(mu, logvar)

        t_p = torch.FloatTensor(pre_state[1].flatten()).to(device)
        final = torch.cat((z, t_p.unsqueeze(0)), 1)
        return final.to(torch.device("cpu")).detach(), reward, done, info


    def reset(self):
        pre_state = self._env.reset()
        self.last_image = pre_state[0]
        obs = torch.Tensor(pre_state[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).float().to(device)
        mu, logvar = self.vae.encode(obs)
        z = self.vae.reparametrize(mu, logvar)

        t_p = torch.FloatTensor(pre_state[1].flatten()).to(device)
        final = torch.cat((z, t_p.unsqueeze(0)), 1)
        #import pudb; pudb.set_trace()
        return final.to(torch.device("cpu")).detach()


    def render(self):
        cv2.imshow("CarlaGymEnv Cam: {}".format(self.rank), self.last_image)
        cv2.waitKey(1)
        #print('Rendering')

    @classmethod
    def fromrank(cls, rank, sparse=True, withlock=False, distl=False):
        env = CarlaGymEnv.CarEnv(rank, render=True, step_type="other", benchmark="STDFixed", auto_reset=False, discrete=False, sparse=sparse, withlock=withlock, dist_reward=distl)

        return cls(env, vae_model, rank=rank)



def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def run():

    parser = argparse.ArgumentParser(description='Carla RL and VAE')

    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--vae', action='store_true', default=False,
                        help='Train and use VAE.')
    parser.add_argument('--epochs', type=int, default=0, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument("--sparse", type=str2bool, nargs='?',
                            const=True, default=False,
                            help="Activate Sparse Rewards.")
    parser.add_argument("--rl", help="Rl algorithm to use",type=str, default='dqn')
    parser.add_argument('--maxram', type=int, default=5, metavar='N',
                        help='Max ram usage for the dataset in GB, its approximate and multiple of batch size, final size is +- 2GB')
    parser.add_argument('--production', action='store_true', default=False,
                        help='Turn this on for a server production run, and default is for local env')

    args = parser.parse_args()


    if(args.production):
        args.batch_size = 1024
        args.epochs = 2000 if args.epochs == 0 else args.epochs 
        args.maxram = 13
    else:
        args.batch_size = 8
        args.epochs = 100 if args.epochs == 0  else args.epochs 
        args.maxram = 7

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print(args)

    device = torch.device("cuda" if args.cuda else "cpu")
    print('Cuda exists: {}, Cuda version {}'.format(torch.cuda.is_available(), torch.version.cuda))
    print('Running Rl algorithm: {}'.format(args.rl))



    files_to_save = []
    base_save_dir = './project/savedmodels/rl/'


    save_dir = base_save_dir+'sac_a3c/'

    hyperps = {}
    hyperps['max_epochs'] = args.epochs
    hyperps['max_steps'] = 10000
    hyperps['log_interval'] = int(args.epochs/10)
    hyperps['maxram'] = args.maxram
    hyperps['q_lr'] = 0.003
    hyperps['a_lr']= 0.003
    hyperps['action_scale'] = 2.5
    hyperps['action_bias'] = 0
    hyperps['hidden'] = 256
    hyperps['rnn_steps'] = 4
    hyperps['batch_size'] = args.batch_size
    hyperps['start_steps'] = args.batch_size*3

    hyperps['updates_per_step'] = 1
    hyperps['update_every'] = 1

    hyperps['alpha'] = 0.1
    hyperps['gamma'] = 0.99
    hyperps['tau'] = 0.005
    hyperps['critic_target_update'] = 4
    hyperps['seq'] = 4
    hyperps['update_step'] = 4
    hyperps['betas'] = (0.9, 0.999)


    hyperps['log_std_max'] = 1
    hyperps['log_std_min'] = 0
    hyperps['epsilon'] = 1e-6
    hyperps['maxmem'] = 2000000

 
    env = CarlaGymEnv.CarEnv(0, render=True, step_type="other", benchmark="STDRandom", auto_reset=False, discrete=False, sparse=args.sparse, dist_reward=True)
    final_nn = sac_simple_channel.run_sac(env, ((300, 900), 3), 2, hyperps)
    #final_nn = model_tester.run_sac(env, ((300, 900), 3), 2, hyperps)
    #final_nn = rl_human.run_human_gathering(env, ((300, 900), 3), 2, hyperps)

    final_pol = 'pol_model_final.tar'
    final_q1 = 'q1_model_final.tar'
    final_q2 = 'q2_model_final.tar'
    final_tq1 = 'tq1_model_final.tar'
    final_tq2 = 'tq2_model_final.tar'

    #data_pol = open(save_dir+final_pol, 'rb').read()
    #data_q1 = open(save_dir+final_q1, 'rb').read()
    #data_q2 = open(save_dir+final_q2, 'rb').read()
    #data_tq1 = open(save_dir+final_tq1, 'rb').read()
    #data_tq2 = open(save_dir+final_tq2, 'rb').read()


if __name__ == '__main__':
    run()