 
import glob
import time
import os
from collections import namedtuple, deque
from random import randrange, uniform

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np

import random

import glob

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import multivariate_normal

from torch.distributions import Categorical, Normal
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchviz import make_dot

import cv2
from PIL import Image, ImageDraw

import wandb

import random


from torch.utils.data.dataset import Dataset


import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


from architectures_singular import ResNetRLGRU, ResNetRLGRUCritic

# default `log_dir` is "runs" - we'll be more specific here

class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model, device):
        self.model = model
        self.gradients = None
        self.device = device
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient


    def hook_input(self, input_tensor):
        def hook_function(grad_in):
            self.gradients = grad_in

        input_tensor.register_hook(hook_function)



    def generate_gradients(self, input_image, target_output, which=0):
        # Forward

        #input_aug = torch.zeros(2, 6).detach().requires_grad_(True)
        input_image = (input_image[0].clone().detach().requires_grad_(True), input_image[1].clone().detach().requires_grad_(True))
        self.hook_input(input_image[0])


        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop

        # Backward pass
        model_output[0].backward(gradient=model_output[0])
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr



def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))


import io

def save_plot_and_get():
    fig.savefig("test.jpg")
    img = cv2.imread("test.jpg")
    return PIL.Image.fromarray(img)

def buffer_plot_and_get():
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return PIL.Image.open(buf)

def from_canvas():
    lst = list(fig.canvas.get_width_height())
    lst.append(3)
    return PIL.Image.fromarray(np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8).reshape(lst))


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def stats(step, all_actions, all_pol_stats, all_stds, all_means, all_rewards, all_scenario_wins_rewards, stats):

    # create 2 kernels

    
    #import pudb; pudb.set_trace()
    #m1 = (1, 1)
    m1 = (all_means[-1][0], all_means[-1][1])

    s1 = np.eye(2)
    s1[0][0] = all_stds[-1][0]
    s1[1][1] = all_stds[-1][1]

    k1 = multivariate_normal(mean=m1, cov=s1)

    #m2 = (1,1)
    #s2 = np.eye(2)
    #k2 = multivariate_normal(mean=m2, cov=s2)

    # create a grid of (x,y) coordinates at which to evaluate the kernels
    xlim = (-2.5, 2.5)
    ylim = (-2.5, 2.5)
    xres = 100
    yres = 100

    x = np.linspace(xlim[0], xlim[1], xres)
    y = np.linspace(ylim[0], ylim[1], yres)
    xx, yy = np.meshgrid(x,y)

    # evaluate kernels at grid points
    xxyy = np.c_[xx.ravel(), yy.ravel()]
    zz = k1.pdf(xxyy) #+ k2.pdf(xxyy)

    # reshape and plot image
    #img = zz.reshape((xres,yres))



    x, y = np.mgrid[-2.5:2.5:.05, -2.5:2.5:.05]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y

    z = k1.pdf(pos)

    fig, ax = plt.subplots()



    ax.contourf(x,y,z)
    ax.plot(all_pol_stats[-1][0], all_pol_stats[-1][1], marker='o', markersize=3, color="red")
    ax.set_xlabel('Throttle')
    ax.set_ylabel('Steering')
    #plt.title('Carla Car Control')
    fig.savefig('./data/gauss/gauss_{}.png'.format(step))

    plt.close(fig)

    #plt.show()

    #plt.imshow(img);#plt.show()


def metrify(obs, steps, wall_start, all_actions, all_pol_stats, all_stds, all_means, all_rewards, all_scenario_wins_rewards, all_final_reward, all_q_vals, to_plot):
    #import pudb; pudb.set_trace()
    m1 = (all_means[-1][0], all_means[-1][1])

    s1 = np.eye(2)
    s1[0][0] = all_stds[-1][0]
    s1[1][1] = all_stds[-1][1]

    k1 = multivariate_normal(mean=m1, cov=s1)

    # create a grid of (x,y) coordinates at which to evaluate the kernels
    xlim = (-2.5, 2.5)
    ylim = (-2.5, 2.5)
    xres = 100
    yres = 100

    x = np.linspace(xlim[0], xlim[1], xres)
    y = np.linspace(ylim[0], ylim[1], yres)
    xx, yy = np.meshgrid(x,y)

    # evaluate kernels at grid points
    xxyy = np.c_[xx.ravel(), yy.ravel()]
    zz = k1.pdf(xxyy) #+ k2.pdf(xxyy)


    x, y = np.mgrid[-2.75:2.75:.05, -2.75:2.75:.05]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y

    z = k1.pdf(pos)

    fig, ax = plt.subplots()

    ax.contourf(x,y,z)
    ax.plot(all_pol_stats[-1][0], all_pol_stats[-1][1], marker='o', markersize=3, color="red")
    ax.set_xlabel('Throttle')
    ax.set_ylabel('Steering')


    pil_plot = fig2img(fig).resize((400, 300), Image.ANTIALIAS)

    plot_img = np.asarray(pil_plot)[...,:3]

    
    pil_obs = transforms.ToPILImage()(obs[0][0].cpu())
    _draw = ImageDraw.Draw(pil_obs)

    _draw.text((5, 10), 'FPS: %.3f, steps: %.3f' % (steps / (time.time() - wall_start), steps))

    _draw.text((5, 30), 'Steer: %.3f' % all_pol_stats[-1][0])

    _draw.text((5, 50), 'Throttle: %.3f' % all_pol_stats[-1][1])



    #import pudb; pudb.set_trace()
    entropy = Normal(torch.FloatTensor(all_means[-1]), torch.FloatTensor(all_stds[-1])).entropy()
    entropy1, entropy2 = entropy[0].item(), entropy[1].item()
    _draw.text((5, 70), 'Entropy: {:.3f}; {:.3f}'.format(entropy1, entropy2))


    _combined = Image.fromarray(np.hstack((plot_img, np.asarray(pil_obs))))
    

    cv2.imshow('Sensors', np.asarray(_combined))

    fig2, ax2 = plt.subplots()

    ax2.plot(np.clip(all_rewards[:, [0]], -1, 1.5), label='Speed')
    ax2.plot(np.clip(all_rewards[:, [1]], -1, 1.5), label='Time')
    ax2.plot(np.clip(all_rewards[:, [2]], -1, 1.5), label='Distance')
    ax2.plot(np.clip(all_rewards[:, [3]], -1, 1.5), label='Collision')
    ax2.plot(np.clip(all_rewards[:, [4]], -1, 1.5), label='Lane')

    ax2.plot(np.clip(all_q_vals, -1, 1.5), label='QVal')

    ax2.plot(np.clip(all_final_reward, -1, 1.5), label='Final R')
    #ax2.plot(all_rewards[:, []], label='Lane   ')


    if(to_plot != []):
        np_arr_to_plot = np.asarray(to_plot)
        ax2.plot(np_arr_to_plot[:, [0]], np_arr_to_plot[:, [1]], marker='o', markersize=3, color="red")
        to_plot = []


    plt.legend()

    ax2.set_xlabel('Step')
    ax2.set_ylabel('Reward')


    pil_plot2 = fig2img(fig2)

    plot_img2 = np.asarray(pil_plot2)[...,:3]

    cv2.imshow('Rewards', cv2.cvtColor(np.asarray(plot_img2), cv2.COLOR_BGR2RGB))


    cv2.waitKey(1)
    #plt.title('Carla Car Control')
    #fig2.savefig('./data/reward/reward_{}.png'.format(total_steps))


    """
    with open('test_all_data_{}.npy'.format(total_steps), 'wb') as f:
        np.save(f, np.asarray(all_actions))
        np.save(f, np_all_rewards)
        np.save(f, np.asarray(all_pol_stats))
        np.save(f, np.asarray(all_stds))
        np.save(f, np.asarray(all_means))
        np.save(f, np.asarray(all_scenario_wins_rewards))
        np.save(f, np.asarray(all_final_rewards))
    """


    fig3, ax3 = plt.subplots()
    ax3.plot(np.cumsum(all_final_reward), label='Cumulative Reward')

    pil_plot3 = fig2img(fig3)

    plot_img3 = np.asarray(pil_plot3)[...,:3]

    cv2.imshow('Cumulative Reward', np.asarray(plot_img3))


    plt.close(fig2)
    plt.close(fig)
    plt.close(fig3)

    if(steps != 0 and steps % 200 == 0):
        all_actions = []
        all_rewards = []
        all_pol_stats = []
        all_stds = []
        all_means = []
        all_scenario_wins_rewards = []
        all_final_reward = []
        all_q_vals = []



class DiskBuffer:

  def __init__(self, max_size, filedir='diskbuffer/'):
      self.max_size = max_size
      self.buffer = deque(maxlen=max_size)
      self.filedir = filedir
      self.dir_name = filedir + str(int(round(time.time() * 1000)))
      os.mkdir(self.dir_name)
      self.seq = 0
      print('Created folder {}, for disk buffer use, max exps: {}'.format(self.dir_name, max_size))

  def push(self, state, hidden, action, reward, next_hidden, next_state, done):
      #experience = (state, hidden, action, np.array([reward]), next_hidden, next_state, done)
      #import pudb; pudb.set_trace()

      image_np = np.asarray(state[0]*255).astype(np.uint8)
      image_aug_np = np.asarray(state[1])

      next_image_np = np.asarray(next_state[0]*255).astype(np.uint8)
      next_image_aug_np = np.asarray(next_state[1])

      with open(self.dir_name + '/exp_{}.npz'.format(self.seq), 'wb') as f:
        np.savez(f, image_np, image_aug_np, hidden.to("cpu").detach().numpy(), action.to("cpu").detach().numpy(), np.array([reward]), next_hidden.to("cpu").detach().numpy(), next_image_np, next_image_aug_np, done)
        self.buffer.append(self.dir_name + '/exp_{}.npz'.format(self.seq))
        self.seq += 1


  def sample(self, batch_size):
      state_batch = []
      action_batch = []
      reward_batch = []
      next_state_batch = []
      done_batch = []
      hidden_batch = []
      next_hidden_batch = []

      batch = random.sample(self.buffer, batch_size)
      
      for experience_file in batch:
          data = np.load(experience_file)
          image_ts, image_aug_ts, hidden, action, reward, next_hidden, next_image_ts, next_image_aug_ts, done = data.values()

          state_batch.append((torch.FloatTensor(image_ts)/255, torch.FloatTensor(image_aug_ts)))
          action_batch.append(torch.FloatTensor(action))
          reward_batch.append(reward)
          next_state_batch.append((torch.FloatTensor(next_image_ts)/255, torch.FloatTensor(next_image_aug_ts)))
          done_batch.append(done)
          hidden_batch.append(torch.FloatTensor(hidden))
          next_hidden_batch.append(torch.FloatTensor(next_hidden))

      return (state_batch, hidden_batch, action_batch, reward_batch, next_hidden_batch, next_state_batch, done_batch)



      #import pudb; pudb.set_trace()

      return (state_batch, hidden_batch, action_batch, reward_batch, next_hidden_batch, next_state_batch, done_batch)

  def __len__(self):
      return len(self.buffer)





class BasicBuffer:

  def __init__(self, max_size):
      self.max_size = max_size
      self.buffer = deque(maxlen=max_size)

  def push(self, state, hidden, action, reward, next_hidden, next_state, done):
      experience = (state, hidden, action, np.array([reward]), next_hidden, next_state, done)
      self.buffer.append(experience)

  def sample(self, batch_size):
      state_batch = []
      action_batch = []
      reward_batch = []
      next_state_batch = []
      done_batch = []
      hidden_batch = []
      next_hidden_batch = []

      batch = random.sample(self.buffer, batch_size)

      for experience in batch:
          state, hidden, action, reward, next_hidden, next_state, done = experience
          state_batch.append(state)
          action_batch.append(action)
          reward_batch.append(reward)
          next_state_batch.append(next_state)
          done_batch.append(done)
          hidden_batch.append(hidden)
          next_hidden_batch.append(next_hidden)


      #import pudb; pudb.set_trace()

      return (state_batch, hidden_batch, action_batch, reward_batch, next_hidden_batch, next_state_batch, done_batch)

  def __len__(self):
      return len(self.buffer)






def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

import glob


class LoadBuffer:

  def __init__(self, save_dir):
      self.files = glob.glob(save_dir+'*.npz')
      self.max_size = len(self.files)
      #self.buffer = deque(maxlen=max_size)

  #def push(self, state, action, reward, next_state, done):
  #    experience = (state, action, np.array([reward]), next_state, done)
  #    self.buffer.append(experience)

  def push(self, state, action, reward, next_state, done):
    print('Expert Buffer, configed to not push anything')

  def sample(self, batch_size):
      state_batch = []
      action_batch = []
      reward_batch = []
      next_state_batch = []
      done_batch = []

      batch = random.sample(self.files, batch_size)
      for experience_file in batch:
          data = np.load(experience_file)
          old_obs_0, old_obs_1, action, rewards, obs_0, obs_1, done, info = data.values()


          state_batch.append((torch.FloatTensor(old_obs_0).unsqueeze(0).transpose(1, 3).transpose(2, 3)/255, torch.FloatTensor(old_obs_1)))
          action_batch.append(torch.FloatTensor(action))
          #0.5, 1, 5, 1, 1
          #reward_batch.append(rewards[0]*0.5 + rewards[1] +  rewards[2]*5 + rewards[3] + rewards[4] + 1.5)
          reward_batch.append(2)
          next_state_batch.append((torch.FloatTensor(obs_0).unsqueeze(0).transpose(1, 3).transpose(2, 3)/255, torch.FloatTensor(obs_1)))
          done_batch.append(done)

      return (state_batch, action_batch, np.expand_dims(reward_batch, axis=1), next_state_batch, done_batch)

  def __len__(self):
      return len(self.files)




class SAC():

  def __init__(self, env, obs_size, num_actions, hyperps, device, train=True):

    self.hyperps = hyperps
    self.env = env
    self.device = device

    self.num_actions = num_actions


    if(len(obs_size) == 1):
        self.obs_state = obs_size[0]
        self.obs_state_size = obs_size[0]
        self.actor = ResNetRLGRU(3, 2, 12).to(device) #ResNetRLGRU(3, 2, 12)(self.obs_state, self.num_actions).to(device) 
    else:
        self.obs_state = obs_size
        self.obs_state_size =  obs_size[0][0] * obs_size[0][1] * obs_size[1]
        self.actor = ResNetRLGRU(3, 2, 12).to(device) #ResNetRLGRU(self.obs_state, self.num_actions).to(device)


    self.critic1 = ResNetRLGRUCritic(3, 2, 12).to(device)
    self.critic2 = ResNetRLGRUCritic(3, 2, 12).to(device)

    self.targ_critic1 = ResNetRLGRUCritic(3, 2, 12).to(device)
    self.targ_critic2 = ResNetRLGRUCritic(3, 2, 12).to(device)

    self.targ_critic1.load_state_dict(self.critic1.state_dict())
    self.targ_critic2.load_state_dict(self.critic2.state_dict())


    if(train):
 
        self.critic_optim = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=self.hyperps['q_lr'])
        self.policy_optim = optim.Adam(self.actor.parameters(), lr=self.hyperps['q_lr'])

        # entropy temperature
        self.alpha = self.hyperps['alpha']
        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.hyperps['a_lr'])
        
    else:
        self.try_load()


  def critic(self, obs, action):
    return self.critic1(obs, action), self.critic2(obs, action)


  def critic_target(self, obs, action):
    return self.targ_critic1(obs, action), self.targ_critic2(obs, action)


  def eval(self):
    self.actor.eval()
    self.critic1.eval()
    self.critic2.eval()



  def train(self):
    self.actor.train()
    self.critic1.train()
    self.critic2.train()


  def sample(self, obs):

    mean, log_std, hidden = self.actor.forward(obs)
    
    std = log_std.exp()
    normal = Normal(mean, std)
    x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
    y_t = torch.tanh(x_t)
    action = y_t * self.hyperps['action_scale'] + self.hyperps['action_bias']
    log_prob = normal.log_prob(x_t)

    # Enforcing Action Bound
    log_prob -= torch.log(self.hyperps['action_scale'] * (1 - y_t.pow(2)) + self.hyperps['epsilon'])
    log_prob = log_prob.sum(1, keepdim=True)

    mean = torch.tanh(mean) * self.hyperps['action_scale'] + self.hyperps['action_bias']

    entropy = normal.entropy()
    entropy1, entropy2 = entropy[0][0].item(), entropy[0][1].item()

    #print('Std: {:2.3f}, {:2.3f}, log_std: {:2.3f},{:2.3f}, entropy:{:2.3f}, {:2.3f}'.format(std[0][0].item(),std[0][1].item(), log_std[0][0].item(), log_std[0][1].item(), entropy1, entropy2))
    return action, log_prob, mean, std, hidden


  def prob_action(self, obs, action_to_calc):
    # returns probability given policy -> π(action|obs)

    mean, log_std, _, _ = self.actor.forward(obs)

    log_std = torch.tanh(log_std)
    log_std = self.hyperps['log_std_min'] + 0.5 * (self.hyperps['log_std_max'] - self.hyperps['log_std_min']) * (log_std + 1)

    std = log_std.exp()
    normal = Normal(mean, std)

    y_t = torch.tanh(action_to_calc)

    action = y_t * self.hyperps['action_scale'] + self.hyperps['action_bias']
    
    log_prob = normal.log_prob(action_to_calc)
    
    # Enforcing Action Bound

    log_prob = log_prob - torch.log(self.hyperps['action_scale'] * (1 - y_t.pow(2)) + self.hyperps['epsilon'])
    log_prob = log_prob.sum(1, keepdim=True)


    return torch.pow(2, log_prob)


  def try_load(self, save_dir='./project/savedmodels/rl/sac/'):
    print('Loading Model')
    paths = glob.glob(save_dir + '*_final.tar') ; step = 0
    #import pudb; pudb.set_trace()
    if len(paths) > 4:
        #ckpts = [int(s.split('.')[-2]) for s in paths]
        #ix = np.argmax(ckpts) ; step = ckpts[ix]
        #step = ckpts[ix]
        #import pudb; pudb.set_trace()
        self.targ_critic2.load_state_dict(torch.load(paths[-5]))
        self.targ_critic1.load_state_dict(torch.load(paths[-3]))
        self.critic2.load_state_dict(torch.load(paths[-4]))
        self.critic1.load_state_dict(torch.load(paths[-2]))
        self.actor.load_state_dict(torch.load(paths[-1]))

    #print("\tno saved models") if step == 0 else print("\tloaded model: {}".format(paths[ix]))
    #return step


    def save_models(self, episode_numb, save_dir='./savedmodels/rl/sac/'):
            
        torch.save(self.actor.state_dict(), save_dir+'pol_model.{:.0f}.tar'.format(episode_numb))
        torch.save(self.critic1.state_dict(), save_dir+'q1_model.{:.0f}.tar'.format(episode_numb))
        torch.save(self.critic2.state_dict(), save_dir+'q2_model.{:.0f}.tar'.format(episode_numb))
        torch.save(self.targ_critic1.state_dict(), save_dir+'tq1_model.{:.0f}.tar'.format(episode_numb))
        torch.save(self.targ_critic2.state_dict(), save_dir+'tq2_model.{:.0f}.tar'.format(episode_numb))
        print('\n\t{:.0f} Epsiodes: saved model\n'.format(episode_numb))


    def save_models_final(self, save_dir='./project/savedmodels/rl/sac/'):
        
        torch.save(self.actor.state_dict(), save_dir+'pol_model_final.tar')
        torch.save(self.critic1.state_dict(), save_dir+'q1_model_final.tar')
        torch.save(self.critic2.state_dict(), save_dir+'q2_model_final.tar')
        torch.save(self.targ_critic1.state_dict(), save_dir+'tq1_model_final.tar')
        torch.save(self.targ_critic2.state_dict(), save_dir+'tq2_model_final.tar')
        print('\nFinal SAC saved model\n')


  def update(self, memory, updates, expert_data=False):
        batch_size=self.hyperps['batch_size']
        gamma=self.hyperps['gamma']
        tau=self.hyperps['tau']


        with torch.autograd.set_detect_anomaly(True):

            mem_state_batch, mem_first_hidden_batch, mem_action_batch, mem_reward_batch, mem_next_hidden_batch, mem_next_state_batch, mem_mask_batch = memory.sample(batch_size)
            #import pudb; pudb.set_trace()
            mem_states_obs_batch, mem_states_aug_batch = zip(*mem_state_batch)
            mem_next_states_obs_batch, mem_next_states_aug_batch = zip(*mem_next_state_batch)

            first_hidden_batch = torch.stack(mem_first_hidden_batch).to(self.device).squeeze(1)

            state_obs_batch = torch.stack(mem_states_obs_batch).to(self.device).squeeze(1)
            state_aug_batch = torch.stack(mem_states_aug_batch).to(self.device).squeeze(1)

            next_mem_batch = torch.stack(mem_next_hidden_batch).to(self.device).squeeze(1)
            next_next_state_obs_batch = torch.stack(mem_next_states_obs_batch).to(self.device).squeeze(1)
            next_next_state_aug_batch = torch.stack(mem_next_states_aug_batch).to(self.device).squeeze(1)
            #action_batch = torch.FloatTensor(np.stack(mem_action_batch)).to(self.device)
            #action_batch = torch.FloatTensor(np.stack(mem_action_batch)).to(self.device)
            action_batch = torch.stack(mem_action_batch).to(self.device)
            reward_batch = torch.FloatTensor(np.stack(mem_reward_batch)).to(self.device)
            mask_batch = torch.FloatTensor(np.stack(mem_mask_batch)).to(self.device).unsqueeze(1)


            #import pudb; pudb.set_trace()
            #next_st = 
            #st = (state_obs_batch, state_aug_batch)

            with torch.no_grad():
                #import pudb; pudb.set_trace()
                next_state_action, next_state_log_pi, _, _, next_hidden= self.sample((next_next_state_obs_batch, next_next_state_aug_batch, next_mem_batch))
                qf1_next_target, qf2_next_target = self.critic_target((next_next_state_obs_batch, next_next_state_aug_batch, next_mem_batch), next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = reward_batch + mask_batch * gamma * (min_qf_next_target)
            
                is_cs = 1

                if(expert_data):
                    prob_beh = self.prob_action((state_obs_batch, state_aug_batch, first_hidden_batch.detach()), action_batch) + self.hyperps['epsilon']
                    prob_expert = torch.ones_like(prob_beh) * 0.90 # Assume expert is almost confident on what it did
                    is_cs = torch.min(torch.ones_like(prob_expert)*1.5, prob_expert/prob_beh)

            qf1, qf2 = self.critic((state_obs_batch.detach(), state_aug_batch.detach(), first_hidden_batch.detach()), action_batch.detach())  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1_loss = F.mse_loss(qf1, next_q_value * is_cs)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2, next_q_value * is_cs)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf_loss = qf1_loss + qf2_loss 
 
            #print('is_cs : {}'.format(is_cs))

            self.critic_optim.zero_grad()
            qf_loss.backward()
            self.critic_optim.step()

            pi, log_pi, _, _, _ = self.sample((state_obs_batch, state_aug_batch, first_hidden_batch))

            qf1_pi, qf2_pi = self.critic((state_obs_batch, state_aug_batch, first_hidden_batch), pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)


            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs


            if updates % self.hyperps['critic_target_update'] == 0:
                soft_update(self.targ_critic1, self.critic1, tau)
                soft_update(self.targ_critic2, self.critic2, tau)


            
            return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()






def range_scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2




def get_saliency(obs, model, action, std, device):

    #import pudb; pudb.set_trace()

    #actor_saliency = score_frame(model, history, frame_ix, radius, density, interp_func=occlude, mode='actor')
    #saliency_on_atari_frame(actor_saliency, frame, fudge_factor=200, channel=2)

    #backprop = Backprop(model)

    #final_saliency = apply_transforms(obs[0])

    van_backprop = VanillaBackprop(model, device)
    grad_img = van_backprop.generate_gradients(obs, action)
    grad_img += van_backprop.generate_gradients(obs, action, which=1)
    #grad_img += grad_img.min() + 0.000001
    #grad_img = (np.transpose(grad_img, (1, 2 ,0))*255).astype(np.uint8)
    #import pudb; pudb.set_trace()
    grad_img = range_scale(grad_img, (0, 255)).astype(np.uint8)

    grad_img = np.transpose(grad_img, (1, 2 ,0))

    cv2.imshow('HeatMap/Saliency Map', grad_img)
    cv2.waitKey(1)


def run_sac(env, obs_state, num_actions, hyperps, device=torch.device("cpu"), render=True, metrified=True, save_dir='./', load_buffer_dir='./human_samples/'):
    # Possible improvements: pre calc gamma of samples, sample weight on loss, curriculum learning, more tweaks to hyperparams :/
    # also add auto batch by cuda mem, attention maybe doubt, model based, vq-vae2

    wandb.init(config=hyperps, force=True)

    mem_max_size = hyperps['maxmem']
    mem_start_thr = 0.1
    mem_train_thr = 0.2

    memory = DiskBuffer(mem_max_size, filedir=load_buffer_dir)

    print('Batch size: {}'.format(hyperps['batch_size']))
    
    sac_agent = SAC(env, obs_state, num_actions, hyperps, device)

    wandb.watch(sac_agent.actor)
    wandb.watch(sac_agent.critic1)
    wandb.watch(sac_agent.critic2)

    wall_start = time.time()

    total_steps = 0
    updates = 0

    log_reward = []
    
    all_actions = []
    all_pol_stats = []
    all_stds = []
    all_means = []

    all_rewards = []
    all_scenario_wins_rewards = []
    all_final_rewards = []
    all_q_vals = []

    to_plot = []

    w_vel, w_t, w_dis, w_col, w_lan, w_waypoint = 0.5, 1, 1, 1, 1, 5

    rewards_weights = [w_vel, w_t, w_dis, w_col, w_lan, w_waypoint]
    change_rate = 0.1

    system_of_eqs = []

    old_hidden = torch.zeros(1, 256).to(device)

    done = False

    for epi in range(hyperps['max_epochs']):
        obs = env.reset()

        print('Len of memory buffer: {}'.format(len(memory)))
        
        old_obs = (torch.Tensor(obs[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).float().to(device), torch.FloatTensor(obs[1]).to(device))
        
        total_steps += 1

        print('Epoch: {}, Max Epochs: {}, max steps: {}'.format(epi, hyperps['max_epochs'], hyperps['max_steps']))
        for step_numb in range(hyperps['max_steps']):

            action, log_prob, mean, std, hidden = None, None, None, None, None
            sac_agent.eval()
            
            if(len(memory) < mem_max_size*mem_start_thr):
                action = torch.rand(1, 2).to(device) * 4 - 2
                log_prob, mean, std, hidden = torch.FloatTensor([[2.7]]), action, torch.FloatTensor([[0.5, 0.5]]), old_hidden
                print('Memory not at {:3.3f}%, action random {}'.format(mem_start_thr*100, action))
            else:
                action, log_prob, mean, std, hidden = sac_agent.sample((old_obs[0], old_obs[1], old_hidden)) 
            

            all_q_vals.append(min(sac_agent.critic((old_obs[0], old_obs[1], old_hidden), action)).cpu().item())

            obs, reward, done, info = env.step(action.cpu().detach().numpy()[0])

            all_rewards.append(reward)

            if(total_steps % 100 == 0):
              print('Reward: Vel: {:.5f}, time: {:.5f}, dis: {:.5f}, col: {:.5f}, lan: {:.5f}, waypoint: {:.5f}'.format(w_vel*reward[0], w_t*reward[1], w_dis*reward[2],  w_col*reward[3], w_lan*reward[4], w_waypoint*reward[5]))
            
            wandb.log({'reward_vel':w_vel*reward[0], 'reward_time':w_t*reward[1], 'reward_dist':w_dis*reward[2], 'reward_col':w_col*reward[3], 'reward_lane':w_lan*reward[4], 'reward_waypoint':w_waypoint*reward[5]})

            reward = (w_vel*reward[0] + w_t*reward[1] + w_dis*reward[2] + w_col*reward[3] + w_lan*reward[4] + w_waypoint*reward[5])/6


            if(total_steps % 100 == 0):
              print('Final Sum Reward: {:.5f}'.format(reward))
              print('Total steps {}, max_steps: {}'.format(total_steps, hyperps['max_steps']))


            all_final_rewards.append(reward)


            wandb.log({"final_r": reward, 'action':action.cpu(), 'log_prob':log_prob.cpu(), 'mean':mean.cpu(), 'std':std.cpu()})

            if(info != None):
                print(info)
           

            all_pol_stats.append([action.cpu().detach().numpy()[0][0], action.cpu().detach().numpy()[0][1], (log_prob[0]).item(), torch.clamp(torch.exp(log_prob[0]), 0, 1.0).item()])
            all_means.append([mean.cpu().detach().numpy()[0][0], mean.cpu().detach().numpy()[0][1]])
            all_stds.append([std.cpu().detach().numpy()[0][0], std.cpu().detach().numpy()[0][1]])

            obs = (torch.Tensor(obs[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).float().to(device), torch.FloatTensor(obs[1]).to(device))

            memory.push((old_obs[0].to("cpu"), old_obs[1].to("cpu")), old_hidden.to("cpu"), action.to("cpu"), reward, hidden.to("cpu"), (obs[0].to("cpu"), obs[1].to("cpu")), done)

            #metrify(obs, total_steps, wall_start, np.asarray(all_actions), np.asarray(all_pol_stats), np.asarray(all_stds), np.asarray(all_means), np.asarray(all_rewards), np.asarray(all_scenario_wins_rewards), np.asarray(all_final_rewards), np.asarray(all_q_vals), to_plot)

            old_obs = obs
            old_hidden = hidden

            total_steps += 1
            log_reward.append(reward)

            if done:
                print('In SAC Done: {}, step : {}, epoch: {}'.format(len(memory), step_numb, epi))

                if(info['scen_sucess'] != None and info['scen_sucess'] == 1):
                    all_scenario_wins_rewards.append(1)
                    wandb.log({"all_scenario_wins_rewards": all_scenario_wins_rewards})
                    to_plot.append([total_steps, info['scen_sucess']*0.999])

                elif (info['scen_sucess'] != None and info['scen_sucess'] == -1):
                    all_scenario_wins_rewards.append(-1)
                    wandb.log({"all_scenario_wins_rewards": all_scenario_wins_rewards})

                break


            to_train_mem = memory
            expert_data = False
              

            if len(to_train_mem) > hyperps['batch_size']*10 and len(memory) > mem_train_thr*mem_max_size:
                #cuda_mem_before_train = open("cuda_mem_before_train_{}.txt".format(total_steps), "w")
                #cuda_mem_before_train.write(torch.cuda.memory_summary())
                #cuda_mem_before_train.close()
                
                sac_agent.train()
                print('Going to train, len of mem: {}'.format(len(memory)))

                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = sac_agent.update(to_train_mem, updates, expert_data)
                
                wandb.log({"critic_1_loss": critic_1_loss, "critic_2_loss": critic_2_loss, "policy_loss": policy_loss, 'ent_loss':ent_loss, 'alpha':alpha})

                updates += 1  

                #cuda_mem_before_train = open("cuda_mem_after_train_{}.txt".format(total_steps), "w")
                #cuda_mem_before_train.write(torch.cuda.memory_summary())
                #cuda_mem_before_train.close()          

            if(total_steps != 0 and total_steps % 10_000 == 0):
                print('Saving')
                torch.save({
                        'steps': total_steps,
                        'model_state_dict': sac_agent.actor.state_dict(),
                        }, save_dir+'sac_model_{}.tar'.format(total_steps))

                torch.save({
                        'steps': total_steps,
                        'model_state_dict': sac_agent.critic1.state_dict(),
                        }, save_dir+'sac_c1_model_{}.tar'.format(total_steps))

                torch.save({
                        'steps': total_steps,
                        'model_state_dict': sac_agent.critic2.state_dict(),
                        }, save_dir+'sac_c2_model_{}.tar'.format(total_steps))

                wandb.save(save_dir+'sac_model_{}.tar'.format(total_steps))
                wandb.save(save_dir+'sac_c1_model_{}.tar'.format(total_steps))
                wandb.save(save_dir+'sac_c2_model_{}.tar'.format(total_steps))


            if(total_steps >= hyperps['max_steps']):
              break
            
        
        if(total_steps >= hyperps['max_steps']):
          break
    
    print('Final Save')
    torch.save({
            'steps': total_steps,
            'model_state_dict': sac_agent.actor.state_dict(),
            }, save_dir+'final_sac_model.tar')

    torch.save({
            'steps': total_steps,
            'model_state_dict': sac_agent.critic1.state_dict(),
            }, save_dir+'final_sac_c1_model.tar')

    torch.save({
            'steps': total_steps,
            'model_state_dict': sac_agent.critic2.state_dict(),
            }, save_dir+'final_sac_c2_model.tar')

    wandb.save(save_dir+'final_sac_model.tar')
    wandb.save(save_dir+'final_sac_c1_model.tar')
    wandb.save(save_dir+'final_sac_c2_model.tar')

    return sac_agent


def only_train(env, obs_state, num_actions, hyperps, device=torch.device("cpu"), render=True, metrified=True, save_dir='./'):

    #memory = BasicBuffer(hyperps['maxmem'])

    wandb.init(config=hyperps)


    expert_memory = LoadBuffer('./human_samples/')

    print('Batch size: {}'.format(hyperps['batch_size']))
    
    sac_agent = SAC(env, obs_state, num_actions, hyperps, device)

    wandb.watch(sac_agent.actor)
    wandb.watch(sac_agent.critic1)
    wandb.watch(sac_agent.critic2)

    wall_start = time.time()

    total_steps = 0
    updates = 0


    for epi in range(hyperps['max_epochs']):

        #old_obs = (torch.Tensor(obs[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).float().to(device), torch.FloatTensor(obs[1]).to(device))
        #obs_t = torch.Tensor(obs).unsqueeze(0).transpose(1, 3).transpose(2, 3).float()
        
        total_steps += 1

        print('Epoch: {}, Max Epochs: {}'.format(epi, hyperps['max_epochs']))
 

        for step_numb in range(hyperps['max_steps']):



            sac_agent.train()
            print('Going to train, step:{}'.format(total_steps))

            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = sac_agent.update(expert_memory, updates)
            
            wandb.log({"critic_1_loss": critic_1_loss, "critic_2_loss": critic_2_loss, "policy_loss": policy_loss, 'ent_loss':ent_loss, 'alpha':alpha})

            updates += 1
            #print('Updated Neural Nets. Losses: critic1:{:.4f}, critic2:{:.4f}, policy_loss:{:.4f}, entropy_loss: {:.4f}, alpha:{:.4f}.'.format(critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha))
        
            total_steps += 1

            if(total_steps != 0 and total_steps % 500 == 0):
                print('Saving')
                torch.save({
                        'steps': total_steps,
                        'model_state_dict': sac_agent.actor.state_dict(),
                        }, save_dir+'sac_model_{}.tar'.format(total_steps))

                torch.save({
                        'steps': total_steps,
                        'model_state_dict': sac_agent.critic1.state_dict(),
                        }, save_dir+'sac_c1_model_{}.tar'.format(total_steps))

                torch.save({
                        'steps': total_steps,
                        'model_state_dict': sac_agent.critic2.state_dict(),
                        }, save_dir+'sac_c2_model_{}.tar'.format(total_steps))

                wandb.save(save_dir+'sac_model_{}.tar'.format(total_steps))
                wandb.save(save_dir+'sac_c1_model_{}.tar'.format(total_steps))
                wandb.save(save_dir+'sac_c2_model_{}.tar'.format(total_steps))
    
    
    #sac_agent.save_models_final()

    torch.save({
            'steps': total_steps,
            'model_state_dict': sac_agent.actor.state_dict(),
            }, save_dir+'final_sac_model.tar')

    torch.save({
            'steps': total_steps,
            'model_state_dict': sac_agent.critic1.state_dict(),
            }, save_dir+'final_sac_c1_model.tar')

    torch.save({
            'steps': total_steps,
            'model_state_dict': sac_agent.critic2.state_dict(),
            }, save_dir+'final_sac_c2_model.tar')

    wandb.save(save_dir+'final_sac_model.tar')
    wandb.save(save_dir+'final_sac_c1_model.tar')
    wandb.save(save_dir+'final_sac_c2_model.tar')

    return sac_agent







class PolicyDataset(Dataset):
    def __init__(self, filelist, batch_size, device):

        self.device = device
        self.filelist = filelist
        self.batch_size = batch_size
        #print('Amount of images of carla env : {}'.format(len(filelist)))


    def __len__(self):
        #return 20 * round((len(self.filelist) - 20)/20)
        return len(self.filelist)


    def __getitem__(self, idx):

        state_batch = []
        action_batch = []

        data = np.load(self.filelist[idx])
        old_obs_0, old_obs_1, action, rewards, obs_0, obs_1, done, info = data.values()

        state_batch.append((torch.FloatTensor(old_obs_0).unsqueeze(0).transpose(1, 3).transpose(2, 3)/255, torch.FloatTensor(old_obs_1)))
          
        action_batch.append(torch.FloatTensor(action))
        #0.5, 1, 5, 1, 1
        #reward_batch.append(rewards[0]*0.5 + rewards[1] +  rewards[2]*5 + rewards[3] + rewards[4] + 1.5)
        #reward_batch.append(2)
        #next_state_batch.append((torch.FloatTensor(obs_0).unsqueeze(0).transpose(1, 3).transpose(2, 3)/255, torch.FloatTensor(obs_1)))
        #done_batch.append(done)

        return (state_batch, action_batch)



class CriticDataset(Dataset):
    def __init__(self, filelist, batch_size, device):

        self.device = device
        self.filelist = filelist
        self.batch_size = batch_size
        #print('Amount of images of carla env : {}'.format(len(filelist)))


    def __len__(self):
        #return 20 * round((len(self.filelist) - 20)/20)
        return len(self.filelist)


    def __getitem__(self, idx):
        state_batch = []
        action_batch = []
        reward_batch = []

        data = np.load(self.filelist[idx])
        old_obs_0, old_obs_1, action, rewards, obs_0, obs_1, done, info = data.values()

        state_batch.append((torch.FloatTensor(old_obs_0).unsqueeze(0).transpose(1, 3).transpose(2, 3)/255, torch.FloatTensor(old_obs_1)))
          
        action_batch.append(torch.FloatTensor(action))
        #0.5, 1, 5, 1, 1
        reward_batch.append(rewards[0]*0.5 + rewards[1] +  rewards[2]*5 + rewards[3] + rewards[4] + 1.5)
        #reward_batch.append(2)
        #next_state_batch.append((torch.FloatTensor(obs_0).unsqueeze(0).transpose(1, 3).transpose(2, 3)/255, torch.FloatTensor(obs_1)))
        #done_batch.append(done)

        return (state_batch, action_batch, reward_batch)



def behavior_cloning(env, obs_state, num_actions, hyperps, device=torch.device("cpu"), load_dir='./', log_step=10, save_dir='./'):

    #memory = BasicBuffer(hyperps['maxmem'])

    wandb.init(config=hyperps)

    print('Batch size: {}'.format(hyperps['batch_size']))

    batch_size = hyperps['batch_size']
    
    sac_agent = SAC(env, obs_state, num_actions, hyperps, device)

    wandb.watch(sac_agent.actor)
    wandb.watch(sac_agent.critic1)
    wandb.watch(sac_agent.critic2)

    wall_start = time.time()

    total_steps = 0
    updates = 0

    critic_net = sac_agent.critic1
    policy_net = sac_agent.actor


    files = glob.glob('./human_samples/' + '*.npz')

    files_train, files_val, files_unseen = files[0:int(len(files)*0.7)], files[int(len(files)*0.7):int(len(files)*0.9)], files[int(len(files)*0.9):len(files)]



    policy_dataset = PolicyDataset(files_train, batch_size, device)

    critic_dataset = CriticDataset(files_train, batch_size, device)


    #import pudb; pudb.set_trace()


    train_loader = torch.utils.data.DataLoader(policy_dataset, batch_size=batch_size, num_workers=4)

    epochs = 100


    criterion = nn.MSELoss()
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

    for epoch in range(epochs):

        train_loss = 0

        for batch_idx, data in enumerate(train_loader):
            #data = load_batch(batch_idx, True)
            #import pudb; pudb.set_trace()
            data_img, data_img_aug = data[0][0]
            data_img = data_img.squeeze(1)

            data_action = data[1][0]

            optimizer.zero_grad()

            #import pudb; pudb.set_trace()

            mean, log_std = policy_net((data_img, data_img_aug))
            
            log_std = torch.tanh(log_std)
            log_std = hyperps['log_std_min'] + 0.5 * (hyperps['log_std_max'] - hyperps['log_std_min']) * (log_std + 1)

            std = log_std.exp()
            normal = Normal(mean, std)
    
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
            y_t = torch.tanh(x_t)

            pol_action = y_t * hyperps['action_scale'] + hyperps['action_bias']
            
            
            #log_prob = normal.log_prob(x_t)
            
            # Enforcing Action Bound

            #log_prob = log_prob - torch.log(self.hyperps['action_scale'] * (1 - y_t.pow(2)) + self.hyperps['epsilon'])
            #log_prob = log_prob.sum(1, keepdim=True)

            #mean = torch.tanh(mean) * self.hyperps['action_scale'] + self.hyperps['action_bias']

            #outputs = torch.cat(mu, log_std, dim=1)

            loss = criterion(pol_action, data_action)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if(batch_idx != 0 and batch_idx % log_step == 0):
                print('Epoch: {}, Train Loss: {}, this batch_loss : {}'.format(epoch, train_loss, loss.item()))

        if(epoch != 0 and epoch % 5 == 0):
            print('Saving')
            torch.save({
                'steps': total_steps,
                'model_state_dict': sac_agent.actor.state_dict(),
                }, save_dir+'sac_model_{}_bl.tar'.format(total_steps))

            wandb.save(save_dir+'sac_model_{}_bl.tar'.format(total_steps))

    torch.save({
            'steps': total_steps,
            'model_state_dict': sac_agent.actor.state_dict(),
            }, save_dir+'bc_final_sac_model.tar')

    torch.save({
            'steps': total_steps,
            'model_state_dict': sac_agent.critic1.state_dict(),
            }, save_dir+'bc_final_sac_c1_model.tar')


    wandb.save(save_dir+'bc_final_sac_model.tar')
    wandb.save(save_dir+'bc_final_sac_c1_model.tar')


    return sac_agent




def double_phase(env, obs_state, num_actions, hyperps, device=torch.device("cpu"), load_dir='./', log_step=10, save_dir='./'):


    wandb.init(config=hyperps)

    print('Batch size: {}'.format(hyperps['batch_size']))

    batch_size = hyperps['batch_size']
    
    sac_agent = SAC(env, obs_state, num_actions, hyperps, device)

    wandb.watch(sac_agent.actor)
    wandb.watch(sac_agent.critic1)
    wandb.watch(sac_agent.critic2)

    wall_start = time.time()

    total_steps = 0
    updates = 0

    critic_net = sac_agent.critic1
    policy_net = sac_agent.actor


    files = glob.glob('./human_samples/' + '*.npz')

    files_train, files_val, files_unseen = files[0:int(len(files)*0.7)], files[int(len(files)*0.7):int(len(files)*0.9)], files[int(len(files)*0.9):len(files)]


    policy_dataset = PolicyDataset(files_train, batch_size, device)

    critic_dataset = CriticDataset(files_train, batch_size, device)

    train_loader = torch.utils.data.DataLoader(policy_dataset, batch_size=batch_size, num_workers=4)

    epochs = 2

    criterion = nn.MSELoss()
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

    for epoch in range(epochs):

        train_loss = 0

        old_hidden = torch.zeros(hyperps['batch_size'], 256).to(device)

        for batch_idx, data in enumerate(train_loader):

            data_img, data_img_aug = data[0][0]
            data_img = data_img.squeeze(1)

            data_action = data[1][0]

            optimizer.zero_grad()

            #import pudb; pudb.set_trace()

            mean, log_std, hidden = policy_net((data_img.to(device), data_img_aug.to(device), old_hidden))
            
            log_std = torch.tanh(log_std)
            log_std = hyperps['log_std_min'] + 0.5 * (hyperps['log_std_max'] - hyperps['log_std_min']) * (log_std + 1)

            std = log_std.exp()
            normal = Normal(mean, std)
    
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
            y_t = torch.tanh(x_t)

            pol_action = y_t * hyperps['action_scale'] + hyperps['action_bias']

            loss = criterion(pol_action, data_action.to(device))
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            old_hidden = hidden #hidden.detach()

            if(batch_idx != 0 and batch_idx % log_step == 0):
                print('Epoch: {}, Train Loss: {}, this batch_loss : {}'.format(epoch, train_loss, loss.item()))


        if(epoch != 0 and epoch % 50 == 0):
            print('Saving')
            torch.save({
                'steps': total_steps,
                'model_state_dict': sac_agent.actor.state_dict(),
                }, save_dir+'sac_model_{}_bl.tar'.format(total_steps))

            wandb.save(save_dir+'sac_model_{}_bl.tar'.format(total_steps))



    wandb.save(save_dir+'bc_final_sac_model.tar')

    memory = BasicBuffer(30)

    expert_memory = LoadBuffer('./human_samples/')

    wall_start = time.time()

    total_steps = 0
    updates = 0

    log_reward = []
    
    all_actions = []
    all_pol_stats = []
    all_stds = []
    all_means = []

    all_rewards = []
    all_scenario_wins_rewards = []
    all_final_rewards = []
    all_q_vals = []

    to_plot = []

    #[array([0.03867785]), array([-1.7760651]), array([0.06253806]), array([-5.08939411]), array([-0.1565633])]
    w_vel, w_t, w_dis, w_col, w_lan, w_waypoint = 0.5, 1, 5, 1, 1, 2

    rewards_weights = [w_vel, w_t, w_dis, w_col, w_lan, w_waypoint]
    change_rate = 0.1

    system_of_eqs = []

    old_hidden = torch.zeros(1, 256).to(device)

    for epi in range(hyperps['max_epochs']):
        obs = env.reset()

        print('Len of memory buffer: {}'.format(len(memory)))
        
        old_obs = (torch.Tensor(obs[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).float().to(device), torch.FloatTensor(obs[1]).to(device))
        #obs_t = torch.Tensor(obs).unsqueeze(0).transpose(1, 3).transpose(2, 3).float()
        
        total_steps += 1

        print('Epoch: {}, Max Epochs: {}'.format(epi, hyperps['max_epochs']))
 

        for step_numb in range(hyperps['max_steps']):


            sac_agent.eval()

            action, log_prob, mean, std, hidden = sac_agent.sample((old_obs[0], old_obs[1], old_hidden)) 

            all_q_vals.append(min(sac_agent.critic((old_obs[0], old_obs[1], old_hidden), action)).cpu().item())

            obs, reward, done, info = env.step(action.cpu().detach().numpy()[0])

            all_rewards.append(reward)

            print('Reward: Vel: {:.5f}, time: {:.5f}, dis: {:.5f}, col: {:.5f}, lan: {:.5f}, waypoint: {:.5f}'.format(w_vel*reward[0], w_t*reward[1], w_dis*reward[2],  w_col*reward[3], w_lan*reward[4], w_waypoint*reward[5]))
            
            wandb.log({'reward_vel':w_vel*reward[0], 'reward_time':w_t*reward[1], 'reward_dist':w_dis*reward[2], 'reward_col':w_col*reward[3], 'reward_lane':w_lan*reward[4], 'reward_waypoint':w_waypoint*reward[5]})

            reward = (w_vel*reward[0] + w_t*reward[1] + w_dis*reward[2] + w_col*reward[3] + w_lan*reward[4] + w_waypoint*reward[5])/6


            print('Final Sum Reward: {:.5f}'.format(reward))

            all_final_rewards.append(reward)

            wandb.log({"final_r": reward, 'action':action.cpu(), 'log_prob':log_prob.cpu(), 'mean':mean.cpu(), 'std':std.cpu()})

            if(info != None):
                print(info)
           
            all_pol_stats.append([action.cpu().detach().numpy()[0][0], action.cpu().detach().numpy()[0][1], (log_prob[0]).item(), torch.clamp(torch.exp(log_prob[0]), 0, 1.0).item()])
            all_means.append([mean.cpu().detach().numpy()[0][0], mean.cpu().detach().numpy()[0][1]])
            all_stds.append([std.cpu().detach().numpy()[0][0], std.cpu().detach().numpy()[0][1]])

            obs = (torch.Tensor(obs[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).float().to(device), torch.FloatTensor(obs[1]).to(device))


            #if(total_steps >= 6500):
            memory.push((old_obs[0].to("cpu"), old_obs[1].to("cpu")), old_hidden.to("cpu"), action, reward, hidden.to("cpu"), (obs[0].to("cpu"), obs[1].to("cpu")), done)

            #if(total_steps % 800 > 700):
            #    get_saliency(obs, sac_agent.actor, action.cpu().detach().numpy()[0], std, device)
            #    metrify(obs, total_steps, wall_start, np.asarray(all_actions), np.asarray(all_pol_stats), np.asarray(all_stds), np.asarray(all_means), np.asarray(all_rewards), np.asarray(all_scenario_wins_rewards), np.asarray(all_final_rewards), np.asarray(all_q_vals), to_plot)
                
            old_obs = obs
            old_hidden = hidden

            total_steps += 1
            log_reward.append(reward)

            if done:
                print('In SAC Done: {}'.format(len(memory)))

                if(info['scen_sucess'] != None and info['scen_sucess'] == 1):
                    all_scenario_wins_rewards.append(1)
                    wandb.log({"all_scenario_wins_rewards": all_scenario_wins_rewards})
                    to_plot.append([total_steps, info['scen_sucess']*0.999])

                elif (info['scen_sucess'] != None and info['scen_sucess'] == -1):
                    all_scenario_wins_rewards.append(-1)
                    wandb.log({"all_scenario_wins_rewards": all_scenario_wins_rewards})
                #{'scen_sucess':1, 'scen_metric':bench_rew}


                break

            #print('Len of Memory: {}, Batch Size: {}'.format(len(memory), hyperps['batch_size']))

            to_train_mem = memory
            expert_data = False
            
            if(total_steps < 10000 and rand.random() < 0.3):
                to_train_mem = expert_memory
                expert_data = True


            if len(to_train_mem) > hyperps['batch_size']:
                sac_agent.train()
                print('Going to train')

                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = sac_agent.update(to_train_mem, updates, expert_data)
                
                wandb.log({"critic_1_loss": critic_1_loss, "critic_2_loss": critic_2_loss, "policy_loss": policy_loss, 'ent_loss':ent_loss, 'alpha':alpha})

                updates += 1
                #print('Updated Neural Nets. Losses: critic1:{:.4f}, critic2:{:.4f}, policy_loss:{:.4f}, entropy_loss: {:.4f}, alpha:{:.4f}.'.format(critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha))
            

            if(total_steps != 0 and total_steps % 50 == 0):
                print('Saving')
                torch.save({
                        'steps': total_steps,
                        'model_state_dict': sac_agent.actor.state_dict(),
                        }, save_dir+'sac_model_{}.tar'.format(total_steps))

                torch.save({
                        'steps': total_steps,
                        'model_state_dict': sac_agent.critic1.state_dict(),
                        }, save_dir+'sac_c1_model_{}.tar'.format(total_steps))

                torch.save({
                        'steps': total_steps,
                        'model_state_dict': sac_agent.critic2.state_dict(),
                        }, save_dir+'sac_c2_model_{}.tar'.format(total_steps))

                wandb.save(save_dir+'sac_model_{}.tar'.format(total_steps))
                wandb.save(save_dir+'sac_c1_model_{}.tar'.format(total_steps))
                wandb.save(save_dir+'sac_c2_model_{}.tar'.format(total_steps))
    


            done |= total_steps == hyperps['max_steps']

    torch.save({
            'steps': total_steps,
            'model_state_dict': sac_agent.actor.state_dict(),
            }, save_dir+'final_sac_model.tar')

    torch.save({
            'steps': total_steps,
            'model_state_dict': sac_agent.critic1.state_dict(),
            }, save_dir+'final_sac_c1_model.tar')

    torch.save({
            'steps': total_steps,
            'model_state_dict': sac_agent.critic2.state_dict(),
            }, save_dir+'final_sac_c2_model.tar')

    wandb.save(save_dir+'final_sac_model.tar')
    wandb.save(save_dir+'final_sac_c1_model.tar')
    wandb.save(save_dir+'final_sac_c2_model.tar')



    return sac_agent
