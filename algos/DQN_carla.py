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

import pandas as pd
from scipy.stats import entropy

hyperparameters = {
    
    "sensor_size": (200, [200, 300, 400, 500]),
    
    "kernel_size":(3, [1, 3, 5, 7]),
    "size_1": (32, [16, 32, 64, 128]),
    "size_2": (64, [16, 32, 64, 128]),
    "size_mem": (64, [16, 32, 64, 128]),
    "conv_channels": (6, [4, 6, 8, 12]),

    "trajectory_len": (4, [3, 4, 6, 10]),

    "mem_size": (1024*2048, [4, 8, 16, 32]),

    "batch_size":  (256, [3, 4, 6, 10]),
    "copy_steps": (15, [5, 10, 15, 25]),
    "gamma": (0.95, [0.95, 0.98, 0.99, 0.999]),
    "epislon": (0.3, [0.3, 0.35, 0.4, 0.45])

}

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class Memory(object):
    """Memory buffer for Experience Replay."""

    def __init__(self, max_size):
        """Initialize a buffer containing max_size experiences."""
        self.buffer = deque(maxlen=max_size)

#    def add(self, *trans):
#        """Add an experience to the buffer."""
#        self.buffer.append(Transition(*trans))


    def add(self, trajectory):
        self.buffer.append(trajectory)

    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        buffer_size = len(self.buffer)
        index = np.random.choice(
            np.arange(buffer_size),
            size=batch_size,
            replace=False
        )

        return [self.buffer[i] for i in index]

    def __len__(self):
        """Interface to access buffer length."""
        return len(self.buffer)


class QNetSimple(nn.Module):

    def __init__(self, state_size, action_size, size_1=32, size_2=64, size_3=128):
        super(QNetSimple, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = nn.Linear(state_size, size_1)

        self.fc2 = nn.Linear(size_1, size_2)

        self.fc3 = nn.Linear(size_2, size_3)

        self.final_layer = nn.Linear(size_3, action_size)


    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))
        x = self.final_layer(x)

        return x

    def val_state(self, state):
        actions = self.forward(state)
        state_vals = actions.max(1)[0]
        return state_vals.unsqueeze(1)



class DQNagent:

    def __init__(self, env, state_size, actions_size, hyperparameters, load=False, render=True, see_input=True, max_steps=2000):

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda")

        self.env = env
        self.max_steps = max_steps
        self.render = render
        self.see_input = see_input
        self.action_size = actions_size

        #self.hidden_gru = torch.torch.randn(self.max_batch_size, self.size_mem)

        kernel_size = hyperparameters["kernel_size"][0]
        size_1 = hyperparameters["size_1"][0]
        size_2 = hyperparameters["size_2"][0]
        size_mem = hyperparameters["size_mem"][0]
        conv_channels = hyperparameters["conv_channels"][0]

        self.trajectory_len = hyperparameters["trajectory_len"][0]
        self.batch_size = hyperparameters["batch_size"][0]
        self.memory_amount = hyperparameters["mem_size"][0]
        self.copy_steps = hyperparameters["copy_steps"][0]
        self.gamma = hyperparameters["gamma"][0]
        self.epislon = 0.3



        #self.qnn = QNet(state_size, actions_size, self.batch_size*self.trajectory_len, conv_channels, kernel_size, size_1, size_2, size_mem).to(self.device)
        #self.target = QNet(state_size, actions_size, self.batch_size*self.trajectory_len, conv_channels, kernel_size, size_1, size_2, size_mem).to(self.device)


        self.qnn = QNetSimple(state_size, actions_size).to(self.device)
        self.target = QNetSimple(state_size, actions_size).to(self.device)


        #if(load):
        #    try_load()

        self.target.load_state_dict(self.qnn.state_dict())
        self.target.eval()

        #import pudb; pudb.set_trace()

        self.memory = Memory(self.memory_amount)

        self.optimizer = optim.Adam(self.qnn.parameters())



    def try_load(self, save_dir='./savedmodels/'):
        print('Loading Model')
        paths = glob.glob(save_dir + '*.tar') ; step = 0
        #import pudb; pudb.set_trace()
        if len(paths) > 1:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts) ; step = ckpts[ix]
            step = ckpts[ix]
            self.qnn.load_state_dict(torch.load(paths[-1]))
            self.target.load_state_dict(torch.load(paths[-2]))

        print("\tno saved models") if step == 0 else print("\tloaded model: {}".format(paths[ix]))
        return step

    def save_models(self, episode_numb, save_dir='./savedmodels/'):
        
        torch.save(self.qnn.state_dict(), save_dir+'dqn_qnn.{:.0f}.tar'.format(episode_numb))
        torch.save(self.target.state_dict(), save_dir+'dqn_target.{:.0f}.tar'.format(episode_numb))
        print('\n\t{:.0f} Epsiodes: saved model\n'.format(episode_numb))


    def save_models_final(self, save_dir='./project/savedmodels/rl/dqn/'):
    
        torch.save(self.qnn.state_dict(), save_dir+'final_qnn_model_final.tar')
        torch.save(self.target.state_dict(), save_dir+'final_target_model_final.tar')
        print('\n\tFinal DQN saved model\n')


    def run(self, max_episodes, save=True, split_input=True):

        all_losses = [0]
        log_actions = np.asarray([])

        results_file = open("dqn_results.txt","w+") 

        for epi in range(max_episodes):
            obs = self.env.reset()
            obs_t = torch.Tensor(obs).unsqueeze(0).type(torch.float32).to(self.device)

            rewards = torch.IntTensor([])

            epi_start = time.time()
            epi_reward = 0


            if(save and epi != 0 and epi%100 == 0):
                self.save_models(epi)

            for step_numb in range(self.max_steps):
                
                trajectory = []

                for trajectory_step in range(self.trajectory_len):

                    #import pudb; pudb.set_trace()

                    actions = self.qnn(obs_t)[0]
                    actions = actions.cpu()
                            
                    #best_action = actions.max(0)[1].clone().cpu().type(torch.uint8).view(-1, 1)

                    #if(uniform(0, 1) < self.epislon):
                    #    print('Exploration Step', end=' ')
                    #    best_action = torch.Tensor([randrange(self.action_size)]).type(torch.uint8).view(-1, 1)

                    best_action = Categorical(F.softmax(actions)).sample().item()

                    old_obs = obs_t.clone()

                    #print('Action: {}'.format(best_action[0].numpy()[0].item()))

                    #log_actions = np.concatenate((log_actions, np.asarray([best_action[0].numpy()[0].item()])))

                    #obs, reward, done, info = self.env.step(best_action[0].numpy()[0].item())


                    print('Action: {}'.format(best_action))

                    log_actions = np.concatenate((log_actions, np.asarray([best_action])))

                    obs, reward, done, info = self.env.step(best_action)

                    epi_reward += reward
 
                    obs_t = torch.Tensor(obs).unsqueeze(0).type(torch.float32).to(self.device)

                    trajectory.append(Transition(old_obs, actions.to(self.device), obs_t, torch.FloatTensor([reward]).view(-1, 1).to(self.device)))


                    if(len(trajectory) == 4):
                        break

                    if(self.render):
                        self.env.render()


                #self.memory.add(old_obs, best_action.to(self.device), obs_t, torch.FloatTensor([reward]).view(-1, 1).to(self.device))

                self.memory.add(trajectory)
                

                if(len(self.memory) >= self.batch_size):
                    tr_loss = self.train_model(step_numb, self.qnn, self.target, self.copy_steps, self.memory, self.optimizer, self.batch_size, self.device, self.gamma)
                    all_losses.append(tr_loss)


                if done:
                    step_time = step_numb/(time.time()- epi_start)
                    if(len(self.memory) >= self.batch_size*2):
                        value, counts = np.unique(log_actions, return_counts=True)

                        print('Episode: {:4d} done with reward: {:2.3f}, steps:{:4d}, steps/sec:{:3.3f}, avg loss: {:3.3f}, last loss{:3.3f}, entropy:{}'.format(epi, epi_reward, step_numb, step_time, sum(all_losses)/len(all_losses), all_losses[-1], entropy(counts)))
                        
                        results_file.write("{}, {}, {}, {}, {}, {}, {}\n".format(epi, epi_reward, step_numb, step_time, sum(all_losses)/len(all_losses), all_losses[-1], entropy(counts)))
                        results_file.flush()
                        print('Len of memory: {}'.format(len(self.memory)))
                    break


    
    def train_model(self, step_numb, model, target, copy_steps, memory, optimizer, batch_size, device, gamma):
        
        transitions_batch = memory.sample(batch_size)

        new_transitions_batch = [item for sublist in transitions_batch for item in sublist]

        #import pudb; pudb.set_trace()
        
        # account for incomplete trajectories
        actual_batch_size = len(new_transitions_batch)      

        batch = Transition(*zip(*new_transitions_batch))


        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])

        
        all_states = torch.cat(batch.state)
        #all_states = all_states.transpose(1,2)
        all_rewards = torch.cat(batch.reward).unsqueeze(1)
        all_actions = torch.cat(batch.action).unsqueeze(1).view(self.batch_size*4, 1, -1)

        
        q_s_a = model(all_states).gather(1, torch.clamp(all_actions, 0, 9).type(torch.long))


        next_state_values = torch.zeros(actual_batch_size, device=device)
        
        next_state_values[non_final_mask] = model(non_final_next_states).squeeze(1).max(1)[0]

        loss_q = 0.5 * (next_state_values.unsqueeze(1) - (all_rewards + gamma*target.val_state(non_final_next_states).detach()))**2
        

        #loss_ent = 

        loss = torch.mean(loss_q)


        torch.autograd.set_detect_anomaly(True)
        optimizer.zero_grad()
        loss.backward()
        
        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        if(step_numb % copy_steps):
            target.load_state_dict(model.state_dict())


        print('Finished training, batch_size was: {:d}'.format(actual_batch_size))

        return loss




def run_dqn(env, obs_state, num_actions, device, hypeparameters, render=False):

    #state_size = ((hyperparameters["sensor_size"][0], hyperparameters["sensor_size"][0]*2), 3)
    #actions_size = 9
    #import pudb; pudb.set_trace()
    dqn_agent = DQNagent(env, obs_state, num_actions, hyperparameters, load=False, render=render)

    dqn_agent.run(hypeparameters['max_epochs'])

    dqn_agent.save_models_final()







"""

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



import glob



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

import wandb


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


    def __len__(self):
      return len(self.states)


class Policy(nn.Module):

    def __init__(self, state_size, action_size, aditional_aug, conv_channels=6, kernel_size=3, size_1=32, size_2=64, size_3=32):
        super(Policy, self).__init__()

        self.state_size, channels_in = state_size
        self.action_size = action_size
        self.aditional_aug = aditional_aug

        #self.max_batch_size = max_batch_size

        self.conv1 = nn.Conv2d(channels_in, conv_channels, kernel_size, stride=1)

        self.size_now = self.conv_output_shape(self.state_size) 

        self.pool1 = nn.MaxPool2d(2, 2)

        self.size_now = (int(self.size_now[0]/2), int(self.size_now[1]/2))

        self.conv2 = nn.Conv2d(conv_channels, conv_channels*2, kernel_size)

        self.size_now = self.conv_output_shape(self.size_now)

        self.pool2 = nn.MaxPool2d(2, 2)

        self.size_now = int(self.size_now[0]/2) * int(self.size_now[1]/2) * conv_channels*2

        self.fc1 = nn.Linear(self.size_now + aditional_aug, size_1)

        self.fc2 = nn.Linear(size_1, size_2)

        self.fc3 = nn.Linear(size_2, size_3)

        self.final_layer = nn.Linear(size_3, action_size)

        self.critic = nn.Linear(size_3, 1)


    def forward(self, x):
        #import pudb; pudb.set_trace()

        x, aditional = x

        x = self.pool1(F.relu(self.conv1(x)))

        x = self.pool2(F.relu(self.conv2(x)))

        
        x = x.reshape(-1, self.size_now)

        #import pudb; pudb.set_trace()
        x = torch.cat((x, aditional.reshape(-1, 12)), dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        action = F.softmax(self.final_layer(x))

        value = F.relu(self.critic(x))

        return Categorical(action), value


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



class PolicySimple(nn.Module):

    def __init__(self, state_size, action_size, max_batch_size, conv_channels=6, kernel_size=3, size_1=32, size_2=64, size_3=32):
        super(PolicySimple, self).__init__()

        self.state_size = 412
        self.action_size = action_size

        self.fc1 = nn.Linear(self.state_size, size_1)

        self.fc2 = nn.Linear(size_1, size_2)

        self.fc3 = nn.Linear(size_2, size_3)

        self.final_layer = nn.Linear(size_3, action_size)

        self.critic = nn.Linear(size_3, 1)


    def forward(self, x):
        #import pudb; pudb.set_trace()


        x = x.reshape(-1, self.state_size)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        action = F.softmax(self.final_layer(x))

        value = F.relu(self.critic(x))

        return Categorical(action), value


def ppo_update(memory, new_policy, old_policy, MseLoss, optimizer, device, hypeparameters):

  lr = hypeparameters['q_lr']
  betas = hypeparameters['betas']
  gamma = hypeparameters['gamma']
  K_epochs = hypeparameters['K_epochs']
  eps_clip = hypeparameters['eps_clip']

  rewards = []
  discounted_reward = 0
  for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
      if is_terminal:
          discounted_reward = 0
      discounted_reward = reward + (gamma * discounted_reward)
      rewards.insert(0, discounted_reward)
  
  # Normalizing the rewards:
  rewards = torch.tensor(rewards).to(device).float()
  rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
  
  # convert list to tensor

  #old_states = torch.stack(memory.states).to(device).detach()
  old_states = list(zip(*memory.states))
  old_actions = torch.stack(memory.actions).to(device).detach()
  old_logprobs = torch.stack(memory.logprobs).to(device).detach()
  
  losses = []
  entropies = []

  #import pudb; pudb.set_trace()

  for _ in range(K_epochs):

      dist_action_wouldtaken, state_values = new_policy((torch.stack(old_states[0]).squeeze(1), torch.stack(old_states[1])))

      logprobs = dist_action_wouldtaken.log_prob(old_actions)
      dist_entropy = dist_action_wouldtaken.entropy()
      state_values = torch.squeeze(state_values)

      #logprobs, state_values, dist_entropy = new_policy.evaluate(old_states, old_actions)
      
      # Finding the ratio (pi_theta / pi_theta__old):
      ratios = torch.exp(logprobs - old_logprobs.detach())
          
      # Finding Surrogate Loss:
      advantages = rewards - state_values.detach()
      surr1 = ratios * advantages
      surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
      loss = -torch.min(surr1, surr2) + 0.5*MseLoss(state_values, rewards) - 0.01*dist_entropy

      # take gradient step
      optimizer.zero_grad()
      loss.mean().backward()
      losses.append(loss.mean())
      entropies.append(dist_entropy.mean())
      optimizer.step()
  
  # Copy new weights into old policy:
  old_policy.load_state_dict(new_policy.state_dict())
  return losses, entropies


def run(env, max_episodes=100, max_steps=100):

  for epi in range(max_episodes):
    epi_reward = 0
    for step in range(max_steps):

      action = env.action_space.sample()
      obs, reward, done, _ = env.step(action)   

      epi_reward += reward
      if(done):
        print('Done episode {:3d}, with total reward: {:2.2f}'.format(epi, epi_reward))
        break



def just_run(policy_nn):
  for epi in range(1000000):
    obs = env.reset()
    for step_numb in range(1000000):
        obs_t = torch.Tensor(obs).unsqueeze(0).transpose(1, 3).transpose(2, 3).to(device)
        dist, val = policy_nn(obs_t)

        action = dist.sample()
        
        obs, reward, done, info = env.step(action.cpu().numpy())

        if done:
            break

      
        env.render()
        cv2.imshow('input', obs)
        cv2.waitKey(1)

#746959


def run_pg(env, obs_state, num_actions, device, hypeparameters, render=False):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ppo_net = PolicySimple(412, 9, 1).to(device).float()
    old_ppo_net = PolicySimple(412, 9, 1).to(device).float()
    old_ppo_net.load_state_dict(ppo_net.state_dict())

    mseloss = nn.MSELoss()
    optimizer = optim.Adam(ppo_net.parameters())


    action = None

    memory = Memory()

    all_episode_rewards = np.asarray([])

    total_steps = 0

    render = True

    for epi in range(hypeparameters['max_epochs']):
        
        obs = env.reset()
        #obs_t = torch.Tensor(obs).unsqueeze(0).transpose(1, 3).transpose(2, 3).to(device).float()
        obs_t = torch.Tensor(obs).to(device).float()

        epi_rewards = 0

        for step_numb in range(hypeparameters['max_steps']):

            #import pudb; pudb.set_trace()

            dist, val = ppo_net(obs_t)

            action = dist.sample()

            obs, reward, done, info = env.step(action.cpu().item())

            epi_rewards += reward

            #obs_t = torch.Tensor(obs[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).to(device).float()
            obs_t = torch.Tensor(obs).to(device).float()

            memory.actions.append(action)
            memory.states.append(obs_t)
            memory.logprobs.append(dist.log_prob(action))
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            total_steps += 1

            if done:
                print('Done epoch {}, max_epochs: {}, len memory: {}'.format(epi, hypeparameters['max_epochs'], len(memory)))
                break

  
            # update if its time
            if total_steps % hypeparameters['update_step'] == 0:
                losses, entropies = ppo_update(memory, ppo_net, old_ppo_net, mseloss, optimizer, device, hypeparameters)
                memory.clear_memory()
                print('Trained Neural Networks')

                print('Lossed mean: {:2.4f}, entropies mean : {:2.4f}, episode reward: {:3.4f}'.format(sum(losses)/len(losses), sum(entropies)/len(entropies), epi_rewards))
            

            #running_reward += reward
            if render:
                env.render()
                

        #if((epi + 1) % 500 == 0):
          #save_models(policy_nn, epi)


  


    stats = None
    return ppo_net, hypeparameters['max_epochs'], stats



def run_ppo_convs(env, device, hypeparameters, save_dir, render=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #wandb.init(config=hypeparameters)

    ppo_net = Policy(((300, 900), 3), 9, 12).to(device).float()

    old_ppo_net = Policy(((300, 900), 3), 9, 12).to(device).float()
    old_ppo_net.load_state_dict(ppo_net.state_dict())

    #wandb.watch(ppo_net)

    mseloss = nn.MSELoss()
    optimizer = optim.Adam(ppo_net.parameters())

    action = None

    memory = Memory()

    all_episode_rewards = np.asarray([])

    total_steps = 0

    render = True

    last_loss = 0

    for epi in range(hypeparameters['max_epochs']):
        
        obs = env.reset()
        obs_t = (torch.Tensor(obs[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).to(device).float(), torch.Tensor(obs[1]).to(device).float())
        #obs_t = torch.Tensor(obs).to(device).float()

        epi_rewards = 0


        for step_numb in range(hypeparameters['max_steps']):

            #import pudb; pudb.set_trace()

            dist, val = ppo_net(obs_t)

            action = dist.sample()

            obs, reward, done, info = env.step(action.cpu().item())

            epi_rewards += reward

            obs_t = (torch.Tensor(obs[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).to(device).float(), torch.Tensor(obs[1]).to(device).float())
            #obs_t = torch.Tensor(obs).to(device).float()

            memory.actions.append(action)
            memory.states.append(obs_t)
            memory.logprobs.append(dist.log_prob(action))
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            total_steps += 1

            if done:
                print('Done epoch {}, max_epochs: {}, len memory: {}'.format(epi, hypeparameters['max_epochs'], len(memory)))
                #wandb.log({'episode_steps':step_numb})
                break

  
            # update if its time
            if total_steps % hypeparameters['update_step'] == 0:
                losses, entropies = ppo_update(memory, ppo_net, old_ppo_net, mseloss, optimizer, device, hypeparameters)
                memory.clear_memory()
                print('Trained Neural Networks')
                #wandb.log({"mean_loss": sum(losses)/len(losses), "mean_entropy": sum(entropies)/len(entropies)}, {"episode_reward": epi_rewards})

                print('Lossed mean: {:2.4f}, entropies mean : {:2.4f}, episode reward: {:3.4f}'.format(sum(losses)/len(losses), sum(entropies)/len(entropies), epi_rewards))
                last_loss = sum(losses)/len(losses)
            
            #running_reward += reward
            #if render:
            #    env.render()

            if (total_steps != 0 and total_steps % 500 == 0):
              env.record_video()



            #if(total_steps == 100):
            #  import pudb; pudb.set_trace()
                

        #if((epi + 1) % 50 == 0):
        #  save_models(policy_nn, epi)
  

    torch.save({
            'steps': total_steps,
            'model_state_dict': ppo_net.state_dict(),
            'loss': last_loss,
            }, save_dir+'final_model.tar')

    #wandb.save(save_dir+'final_model.tar')

    return ppo_net, last_loss



import cv2

def run_scenario_convs(env, path_to_nn, render=False):

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  checkpoint = torch.load(path_to_nn)

  ppo_net = Policy(((300, 900), 3), 9, 12).to(device).float()
  ppo_net.load_state_dict(checkpoint['model_state_dict'])

  policy = ppo_net


  obs = env.reset()
  obs_t = (torch.Tensor(obs[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).to(device).float(), torch.Tensor(obs[1]).to(device).float())
  #obs_t = torch.Tensor(obs).to(device).float()



  #writer = cv2.VideoWriter('test1.avi', cv2.VideoWriter_fourcc(*'PIM1'), 25, (900, 300), False)

  # Define the codec and create VideoWriter object
  #fourcc = cv2.VideoWriter_fourcc(*'DIVX')

  #out = cv2.VideoWriter('output.avi', -1, 20.0, (900,300))

  fourcc = cv2.VideoWriter_fourcc(*'MP42')
  video = cv2.VideoWriter('./test.avi', fourcc, float(20), (900, 300))


  while True:

    dist, val = policy(obs_t)

    action = dist.sample()

    action = int(action/3), action%3

    thrt_action, steer_action = action
    #Discrete(3) -> 0, 1, 2 -> transform to -1, 0, 1
    thrt_action -= 1
    steer_action -= 1



    obs, reward, done, info = env.step((torch.FloatTensor([thrt_action]), steer_action.cpu()))


    #render = True
    #if render:
    #  cv2.imshow('rl_agent_sensors', obs[0])
    #  cv2.waitKey(1)

    frame_np = obs[0]*255
    frame_np = frame_np.astype(np.uint8)
    video.write(frame_np)

    obs_t = (torch.Tensor(obs[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).to(device).float(), torch.Tensor(obs[1]).to(device).float())


  # Release everything if job is finished

  video.release()

"""