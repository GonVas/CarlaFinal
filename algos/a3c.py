from __future__ import print_function
import torch, os, time, glob, argparse, sys
import numpy as np
from scipy.signal import lfilter

import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
os.environ['OMP_NUM_THREADS'] = '1'

#from CarlaEnvAuto import CarEnv
"""
try:
    carla_dir = os.environ['CARLA_DIR']

    sys.path.append(glob.glob(carla_dir + '/PythonAPI' + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


import carla
"""

class Policy(nn.Module):

    def __init__(self, state_size, action_size, max_batch_size, conv_channels=6, kernel_size=3, size_1=32, size_2=64, size_3=32):
        super(Policy, self).__init__()

        self.state_size = 412
        self.action_size = action_size

        self.fc1 = nn.Linear(self.state_size, size_1)

        self.fc2 = nn.Linear(size_1, size_2)

        self.fc3 = nn.Linear(size_2, size_3)

        self.final_layer = nn.Linear(size_3, action_size)

        self.critic = nn.Linear(size_3, 1)


    def forward(self, x):
        #import pudb; pudb.set_trace()
        x, hx = inputs
        x = x.reshape(-1, self.state_size)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        action = F.softmax(self.final_layer(x))

        value = F.relu(self.critic(x))

        #return Categorical(action), value
        return value, Categorical(action)

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--processes', default=1, type=int, help='number of processes to train with')
    parser.add_argument('--render', default=False, type=bool, help='renders the atari environment')
    parser.add_argument('--test', default=False, type=bool, help='sets lr=0, chooses most likely actions')
    parser.add_argument('--rnn_steps', default=20, type=int, help='steps to train LSTM over')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--seed', default=1, type=int, help='seed random # generators (for reproducibility)')
    parser.add_argument('--gamma', default=0.99, type=float, help='rewards discount factor')
    parser.add_argument('--tau', default=1.0, type=float, help='generalized advantage estimation discount')
    parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
    parser.add_argument('--hidden', default=256, type=int, help='hidden size of GRU')
    parser.add_argument('--cam_width', default=640, type=int, help='hidden size of GRU')
    parser.add_argument('--cam_height', default=480, type=int, help='hidden size of GRU')
    return parser.parse_args()

discount = lambda x, gamma: lfilter([1],[1,-gamma],x[::-1])[::-1] # discounted rewards one liner
#prepro = lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.

def prepro(img):
    return img.astype(np.float32).reshape(3, 640, 480)/255.


#def printlog(args, s, end='\n', mode='a'):
#    print(s, end=end) ; f=open(args.save_dir+'log.txt',mode) ; f.write(s+'\n') ; f.close()

class NNPolicy(nn.Module): # an actor-critic neural network
    def __init__(self, channels, memsize, num_actions):
        super(NNPolicy, self).__init__()
        # 3*640*480
        #self.conv1 = nn.Conv2d(channels, 32, 3, stride=1, padding=1)
        # 32*640*480 
        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 32*320*240
        #self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        # 64*320*240 (This will grab higher features)
        #self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 64*160*120 (This will grab higher features)
        #self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        # 64*160*120 (This will grab higher features)
        #self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 64*80*60 (This will grab higher features)

        self.ln1 = nn.Linear(412, 64)
        self.ln2 = nn.Linear(64, 32)

        self.gru = nn.GRUCell(32, memsize)
        #input_dim = 5
        #hidden_dim = 10
        #n_layers = 1 
        #self.lstm = nn.LSTM(64*80*60, memsize, n_layers, batch_first=True)
        
        self.critic_linear, self.actor_linear = nn.Linear(memsize, 1), nn.Linear(memsize, num_actions)

    def forward(self, inputs, train=True, hard=False):
        inputs, hx = inputs


        #print('Xfinal: size: {}'.format(x.size()))
        #hx = self.lstm(x.view(-1, 64*80*60), (hx))
        #self.lstm(x.view(-1, 64*80*60))

        x = F.elu(self.ln1(inputs))
        x = F.elu(self.ln2(x))

        hx = self.gru(x.view(-1, 32), (hx))
        return self.critic_linear(hx), self.actor_linear(hx), hx

    def try_load(self, save_dir):
        paths = glob.glob(save_dir + '*.tar') ; step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts) ; step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step == 0 else print("\tloaded model: {}".format(paths[ix]))
        return step

class SharedAdam(torch.optim.Adam): # extend a pytorch optimizer so it shares grads across processes
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    self.state[p]['shared_steps'] += 1
                    self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1 # a "step += 1"  comes later
            super.step(closure)


def cost_func(values, logps, actions, rewards):
    np_values = values.view(-1).data.numpy()

    # generalized advantage estimation using \delta_t residuals (a policy gradient method)
    delta_t = np.asarray(rewards) + 0.99 * np_values[1:] - np_values[:-1]
    logpys = logps.gather(1, torch.tensor(actions).view(-1,1))
    gen_adv_est = discount(delta_t, 0.99)
    policy_loss = -(logpys.view(-1) * torch.FloatTensor(gen_adv_est.copy())).sum()
    
    # l2 loss over value estimator
    rewards[-1] += 0.99 * np_values[-1]
    discounted_r = discount(np.asarray(rewards), 0.99)
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
    value_loss = .5 * (discounted_r - values[:-1,0]).pow(2).sum()

    entropy_loss = (-logps * torch.exp(logps)).sum() # entropy definition, for entropy regularization
    return policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

# Hyperps

# hyperps['hidden'] = 256
# hyperps['rnn_steps'] = 6

def train(env, shared_model, shared_optimizer, rank, info, hyperps):

    print('Training')

    env = env.fromrank(0)
   
    model = NNPolicy(channels=3, memsize=hyperps['hidden'], num_actions=2) # a local/unshared model
    state = torch.tensor(env.reset()) # get first state

    start_time = last_disp_time = time.time()
    episode_length, epr, eploss, done  = 0, 0, 0, True # bookkeeping

    while info['frames'][0] <= 8e7: # openai baselines uses 40M frames...we'll use 80M
        model.load_state_dict(shared_model.state_dict()) # sync with shared model

        hx = torch.zeros(1, 256) if done else hx.detach()  # rnn activation vector
        values, logps, actions, rewards = [], [], [], [] # save values for computing gradientss

        for step in range(hyperps['rnn_steps']):
            episode_length += 1
            value, logit, hx = model((state.view(1, 412), hx))
            logp = F.log_softmax(logit, dim=-1)

            action = torch.exp(logp).multinomial(num_samples=1).data[0]#logp.max(1)[1].data if args.test else
            #import pudb; pudb.set_trace()
            state, reward, done, _ = env.step(np.random.rand(2, 1))
            
            #if args.render: env.render()

            state = torch.tensor(state) ; epr += reward
            reward = np.clip(reward, -1, 1) # reward
            done = done or episode_length >= hyperps['max_steps'] # don't playing one ep for too long
            
            #info['frames'].add_(1) ; num_frames = int(info['frames'].item())
            
            #if num_frames % 2e4 == 0: # save every 20K frames
            #    printlog(args, '\n\t{:.0f}K frames: saved model\n'.format(num_frames/1e5))
            #    torch.save(shared_model.state_dict(), args.save_dir+'model.{:.0f}.tar'.format(num_frames/1e5))

            #if done: # update shared data
                #info['episodes'] += 1
                #interp = 1 if info['episodes'][0] == 1 else 1 - args.horizon
                #info['run_epr'].mul_(1-interp).add_(interp * epr)
                #info['run_loss'].mul_(1-interp).add_(interp * eploss)

            #if rank == 0 and time.time() - last_disp_time > 60: # print info ~ every minute
            #    elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
            #    printlog(args, '\n Torch A3C Agent Metrics: time {}, episodes {:.0f}, frames {:.1f}M, mean epr {:.2f}, run loss {:.2f}'
            #        .format(elapsed, info['episodes'].item(), num_frames/1e6,
            #        info['run_epr'].item(), info['run_loss'].item()))
            #    last_disp_time = time.time()

            if done: # maybe print info.
                episode_length, epr, eploss = 0, 0, 0
                state = torch.tensor(env.reset())

            values.append(value) ; logps.append(logp) ; actions.append(action) ; rewards.append(reward)

        next_value = torch.zeros(1,1) if done else model((state.unsqueeze(0), hx))[0]
        values.append(next_value.detach())

        loss = cost_func(torch.cat(values), torch.cat(logps), torch.cat(actions), np.asarray(rewards))
        eploss += loss.item()
        shared_optimizer.zero_grad() ; loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)
        
        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is None: shared_param._grad = param.grad # sync gradients with shared model
        shared_optimizer.step()

    print('Finished Training')



def run_a3c(env, obs_state, num_actions, hyperps, save_dir, device=torch.device("cpu"), render=False):
    import pudb; pudb.set_trace()
    mp.set_start_method('spawn')

    shared_model = NNPolicy(channels=3, memsize=256, num_actions=2).share_memory()
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=hyperps['q_lr'])

    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames']}
    info['frames'] += shared_model.try_load(save_dir) * 1e6
    #if int(info['frames'].item()) == 0: printlog('ola','', end='', mode='w') # clear log file
    
    processes = []

    print('Spwaning training processes')

    #for rank in range(args.processes):
    #    p = mp.Process(target=train, args=(env, shared_model, shared_optimizer, rank, info, hyperps))
    #    p.start() ; processes.append(p)
    #for p in processes: p.join()

    train(env, shared_model, shared_optimizer, 0, info, hyperps)
