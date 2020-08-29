import glob
import time
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
from architectures import ResNetRLGRU, ResNetRLGRUCritic

# default `log_dir` 





class SAC():

  def __init__(self, rank, env_action_shape, hyperps, device, train=True):
    self.hyperps = hyperps
    self.env_action_shape = env_action_shape
    self.device = device
    
    #self.num_actions = num_actions


    #if(len(obs_size) == 1):
        #self.obs_state = obs_size[0]
        #self.obs_state_size = obs_size[0]
    
    self.actor = ResNetRLGRU(3, 2, 12).to(device) #ResNetRLGRU(3, 2, 12)(self.obs_state, self.num_actions).to(device) 
    #else:
        #self.obs_state = obs_size
        #self.obs_state_size =  obs_size[0][0] * obs_size[0][1] * obs_size[1]
    #    self.actor = ResNetRLGRU(3, 2, 12).to(device) #ResNetRLGRU(self.obs_state, self.num_actions).to(device)


    self.critic1 = ResNetRLGRUCritic(3, 2, 12).to(device)
    self.critic2 = ResNetRLGRUCritic(3, 2, 12).to(device)

    self.targ_critic1 = ResNetRLGRUCritic(3, 2, 12).to(device)
    self.targ_critic2 = ResNetRLGRUCritic(3, 2, 12).to(device)

    print('Before params copy')

    params1 = self.critic1.named_parameters()
    params2 = self.targ_critic1.named_parameters()

    dict_params2 = dict(params2)

    for name1, param1 in params1:
        if name1 in dict_params2:
          dict_params2[name1].data.copy_(param1.data)



    params1 = self.critic2.named_parameters()
    params2 = self.targ_critic2.named_parameters()

    dict_params2 = dict(params2)

    for name1, param1 in params1:
        if name1 in dict_params2:
          dict_params2[name1].data.copy_(param1.data)

    print('Afeter params copy')

    if(train):
 
        self.critic_optim = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=self.hyperps['q_lr'])
        self.policy_optim = optim.Adam(self.actor.parameters(), lr=self.hyperps['q_lr'])

        # entropy temperature
        self.alpha = self.hyperps['alpha']
        #self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
        self.target_entropy = -torch.prod(torch.Tensor(self.env_action_shape).to(self.device)).item()
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

    mean, log_std, hidden, msg = self.actor.forward(obs)
    
    print('Mean : ' + str(mean))

    std = log_std.exp()
    std = torch.clamp(std, 0.01, 0.1)
    #print('Clamped')
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
    return action, log_prob, mean, std, hidden, msg









  def load_models(self, files):
    self.targ_critic2.load_state_dict(torch.load(files[2])['model_state_dict'])
    self.targ_critic1.load_state_dict(torch.load(files[1])['model_state_dict'])
    self.critic2.load_state_dict(torch.load(files[2])['model_state_dict'])
    self.critic1.load_state_dict(torch.load(files[1])['model_state_dict'])
    self.actor.load_state_dict(torch.load(files[0])['model_state_dict'])


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




def metrify(obs, steps, wall_start, all_actions, all_pol_stats, all_stds, all_means, all_rewards, all_scenario_wins_rewards, all_final_reward, all_q_vals, to_plot):

    m1 = (all_means[-1][0], all_means[-1][1])

    s1 = np.eye(2)
    s1[0][0] = all_stds[-1][0]
    s1[1][1] = all_stds[-1][1]

    k1 = multivariate_normal(mean=m1, cov=s1)

    #import pudb; pudb.set_trace()
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
    z_th = np.tanh(z)

    fig, ax = plt.subplots()



    

    ax.contourf(x,y,z_th)
    ax.plot(all_pol_stats[-1][0], all_pol_stats[-1][1], marker='o', markersize=3, color="red")
    ax.set_xlabel('Throttle')
    ax.set_ylabel('Steering')


    pil_plot = fig2img(fig).resize((400, 300), Image.ANTIALIAS)

    plot_img = np.asarray(pil_plot)[...,:3]

    
    pil_obs = transforms.ToPILImage()(obs[0][0])
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







def run_sac(env, obs_state, num_actions, hyperps, device=torch.device("cpu"), render=True, metrified=True):

    #memory = BasicBuffer(hyperps['maxmem'])

    #wandb.init(config=hyperps)

    #wandb.init(project="offline-demo")

    #memory = BasicBuffer(30)

    #expert_memory = LoadBuffer('./human_samples/')

    #print('Batch size: {}'.format(hyperps['batch_size']))

    #import pudb; pudb.set_trace()
    
    #load_files = ['/home/gonvas/Programming/carlaFinal/bc_final_sac_model.tar', '/home/gonvas/Programming/carlaFinal/sac_c1_model_6000.tar', '/home/gonvas/Programming/carlaFinal/sac_c2_model_6000.tar']
    load_files = ['/home/gonvas/Programming/carlaFinal/sac_model_70000.tar', '/home/gonvas/Programming/carlaFinal/sac_c2_model_70000.tar', '/home/gonvas/Programming/carlaFinal/sac_c2_model_70000.tar']

    sac_agent = SAC(0, env.action_space.shape, hyperps, device)


    sac_agent.actor.load_state_dict(torch.load(load_files[0])['model_state_dict'])


    #sac_agent.critic1.encoder.load_state_dict(sac_agent.actor.encoder.state_dict()) 
    #sac_agent.critic1.avg.load_state_dict(sac_agent.actor.avg.state_dict()) 
    #sac_agent.critic1.msg_lin.load_state_dict(sac_agent.actor.msg_lin.state_dict())  

    #critic = 

    #sac_agent.load_models(load_files)

    #wandb.watch(sac_agent.actor)
    #wandb.watch(sac_agent.critic1)
    #wandb.watch(sac_agent.critic2)

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

    final_done = False


    for epi in range(hyperps['max_epochs']):
        obs = env.reset()

        old_obs = (torch.Tensor(obs[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).float().to(device), torch.FloatTensor(obs[1]).to(device))
        
        total_steps += 1

        print('Epoch: {}, Max Epochs: {}'.format(epi, hyperps['max_epochs']))

        for step_numb in range(hyperps['max_steps']):

            sac_agent.eval()

            action, log_prob, mean, std, hidden, _ = sac_agent.sample((old_obs[0], old_obs[1], old_hidden)) 

            #all_q_vals.append(min(sac_agent.critic((old_obs[0], old_obs[1], old_hidden), action)).cpu().item())

            obs, reward, done, info = env.step(action.cpu().detach().numpy()[0])

            all_rewards.append(reward)

            print('Reward: Vel: {:.5f}, time: {:.5f}, dis: {:.5f}, col: {:.5f}, lan: {:.5f}, waypoint: {:.5f}'.format(w_vel*reward[0], w_t*reward[1], w_dis*reward[2],  w_col*reward[3], w_lan*reward[4], w_waypoint*reward[5]))
            
            #wandb.log({'reward_vel':w_vel*reward[0], 'reward_time':w_t*reward[1], 'reward_dist':w_dis*reward[2], 'reward_col':w_col*reward[3], 'reward_lane':w_lan*reward[4], 'reward_waypoint':w_waypoint*reward[5]})

            reward = (w_vel*reward[0] + w_t*reward[1] + w_dis*reward[2] + w_col*reward[3] + w_lan*reward[4] + w_waypoint*reward[5])/6

            print('Final Sum Reward: {:.5f}'.format(reward))

            all_final_rewards.append(reward)


            if(info != None):
                print(info)
           
            all_pol_stats.append([action.cpu().detach().numpy()[0][0], action.cpu().detach().numpy()[0][1], (log_prob[0]).item(), torch.clamp(torch.exp(log_prob[0]), 0, 1.0).item()])
            all_means.append([mean.cpu().detach().numpy()[0][0], mean.cpu().detach().numpy()[0][1]])
            all_stds.append([std.cpu().detach().numpy()[0][0], std.cpu().detach().numpy()[0][1]])
            
            obs = (torch.Tensor(obs[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).float().to(device), torch.FloatTensor(obs[1]).to(device))


            #if(total_steps >= 6500):
            #memory.push((old_obs[0].to("cpu"), old_obs[1].to("cpu")), old_hidden.to("cpu"), action, reward, hidden.to("cpu"), (obs[0].to("cpu"), obs[1].to("cpu")), done)

            #if(total_steps % 800 > 700):
            #    get_saliency(obs, sac_agent.actor, action.cpu().detach().numpy()[0], std, device)
            metrify(obs, total_steps, wall_start, np.asarray(all_actions), np.asarray(all_pol_stats), np.asarray(all_stds), np.asarray(all_means), np.asarray(all_rewards), np.asarray(all_scenario_wins_rewards), np.asarray(all_final_rewards), np.asarray(all_q_vals), to_plot)
                
            old_obs = obs
            old_hidden = hidden

            total_steps += 1
            log_reward.append(reward)

            if done:
                #print('In SAC Done: {}'.format(len(memory)))

                if(info['scen_sucess'] != None and info['scen_sucess'] == 1):
                    all_scenario_wins_rewards.append(1)
                    to_plot.append([total_steps, info['scen_sucess']*0.999])

                elif (info['scen_sucess'] != None and info['scen_sucess'] == -1):
                    all_scenario_wins_rewards.append(-1)
                #{'scen_sucess':1, 'scen_metric':bench_rew}


                try:
                    f = open('done.txt', "r")
                    f.close()
                    print('Model Tester, done everython, ending and saving')
                    final_done = True
                except IOError:
                    pass


                break


            done |= total_steps >= hyperps['max_steps']

        if(final_done):
            print('Saving nets, ending everything')
            break


    return sac_agent

