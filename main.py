import gym
import json
import math
import numpy as np
import pandas as pd
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import random

#Torch and Baseline Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_, clip_grad_norm_
from stable_baselines.common.vec_env import DummyVecEnv
from TradeEnv import StockTradingEnv
from models import ActorNetwork, Order Network
from attention import ScaledDotAttention

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class ReplayBuffer:
  def __init__(self,capacity): 
    self.capacity = capacity
    self.buffer = []
    self.position = 0

  def push(self, state,state_order,act,dec,reward,next_state,done):
      state      = state.detach().cpu().numpy()
      next_state = next_state
      state_order = state_order.detach().cpu().numpy()
      act = act.detach().cpu().numpy()
      self.buffer.append((state,state_order,act,dec,reward,next_state,done))
  
  def sample(self, batch_size):
      state,state_order,act,dec,reward,next_state,done = zip(*random.sample(self.buffer, batch_size))
      return np.concatenate(state), np.concatenate(state_order), np.concatenate(act), dec, reward, np.concatenate(next_state), done
  
  def __len__(self):
      return len(self.buffer)

ticker = 'NVDA'
# data = yf.download(ticker, '2020-01-08', '2020-03-05', interval='2m')
# data = data.to_csv('./data.csv')
data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Code/data/'+ticker+'.csv')
print(data.shape)
env = DummyVecEnv([lambda: StockTradingEnv(data)])

def compute_td_loss(batch_size):

    state,state_order,act,dec,reward,next_state,done = replay_buffer.sample(batch_size)
    
    action_act = torch.LongTensor(act).to(device)
    action_order = torch.LongTensor(dec).unsqueeze(1).to(device)
    state      = torch.FloatTensor(np.float32(state)).to(device)
    state_order      = torch.FloatTensor(state_order).to(device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(device)
    reward     = torch.FloatTensor(reward).to(device)
    done       = torch.FloatTensor(done).to(device)
    
    criterion = nn.SmoothL1Loss(reduction='none')

    # Manager(Actor) Network
    q_values_act,_      = policy_net_act(state)
    next_q_values_act,_ = policy_net_act(next_state)
    q_value_act          = q_values_act.gather(1, action_act).squeeze(1)
    next_q_value_act     = next_q_values_act#.max(1)[0]
    norm = max(rewards) if rewards!=[] else 1
    norm = torch.FloatTensor(norm).to(device)
    expected_q_value_act = (reward/norm) + gamma * next_q_value_act * (1 - done)
    act_loss = criterion(q_value_act,Variable(expected_q_value_act.data))
    act_loss = torch.sum(torch.mean(act_loss,0))
#     act_loss = (q_value_act - Variable(expected_q_value_act.data)).pow(2).mean() 
#     optimizer_act.zero_grad()
#     act_loss.backward()
#     optimizer_act.step()

    # Order Network
    q_values_order,_, _,_      = policy_net_order(state_order)
    q_value_act = torch.unsqueeze(q_value_act,0).permute(1,0,2)
    next_state_order = torch.cat([next_state,q_value_act],1).to(device)
    next_q_values_order,_, _, _ = policy_net_order(next_state_order)
    next_q_value_order     = next_q_values_order.unsqueeze(1).max(1)[0].type(torch.FloatTensor).to(device)
    expected_q_value_order = (reward/norm) + gamma * next_q_value_order * (1 - done) 
    order_loss = criterion(q_values_order,Variable(expected_q_value_order.data))
    order_loss = torch.sum(torch.mean(order_loss,0))
#     order_loss = (torch.FloatTensor(q_values_order) - torch.FloatTensor(expected_q_value_order)).pow(2).mean()
    total_loss = 0.5*act_loss + 0.5*order_loss
    optimizer_order.zero_grad()
    total_loss.backward()
    clip_grad_norm_(policy_net_act.parameters(), 2)
    clip_grad_norm_(policy_net_order.parameters(), 2)
    optimizer_order.step()
 
#     total_loss = 0.3*act_loss + 0.7*order_loss
    
    return order_loss.detach().cpu().numpy(), act_loss.detach().cpu().numpy(), total_loss.detach().cpu().numpy()

replay_initial = 100
replay_buffer = ReplayBuffer(100000)

obs = env.reset()
state_dim_act = obs.shape[2]
state_dim_order = obs.shape[2]
action_dim = 2
print('Launching Environment...')
batch_size = 32
hidden_dim = 64

policy_net_act = ActorNetwork(state_dim_act,hidden_dim).to(device)
policy_net_order = OrderNetwork(state_dim_order,action_dim,hidden_dim).to(device)
policy_lr = 1e-3
optimizer_order = optim.Adam(policy_net_order.parameters(),lr=policy_lr)

ord_l = None
act_l = None
TD_Loss = None
load_model = False 
if load_model==True:
  #Load Actor Policy Net
    policy_checkpoint = torch.load(checkpoint_name+'/policy_net_act.pth.tar',map_location='cpu') 
    policy_net_act.load_state_dict(policy_checkpoint['model_state_dict'])
    optimizer_act.load_state_dict(policy_checkpoint['optimizer_state_dict'])
    TD_Loss = policy_checkpoint['loss']
  #Load Order Policy Net
    policy_checkpoint = torch.load(checkpoint_name+'/policy_net_order.pth.tar',map_location='cpu') 
    policy_net_order.load_state_dict(policy_checkpoint['model_state_dict'])
    optimizer_order.load_state_dict(policy_checkpoint['optimizer_state_dict'])
    TD_Loss = policy_checkpoint['loss']

num_frames  = 100000 #Steps
weights_act = [];weights_ord = []
rewards = [];loss_order = [];loss_actor = [];loss = [];profit = [];forecast = [];action_list = []
gamma = 0.99

epsilon_start = 0.90
epsilon_final = 0.01
epsilon_decay = 30000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

print('Training Started...')
print('-'*100)
state = obs

for frame_idx in range(1, num_frames + 1):
    epsilon = epsilon_by_frame(frame_idx)
    state = torch.FloatTensor(state).to(device)
    act,w_act = policy_net_act.act(state, epsilon)
    forecast.append(act)
    state_order = torch.cat([state,torch.unsqueeze(act,0)],1).to(device)
    order,dec,w_ord = policy_net_order.act(state_order, epsilon)
    action_list.append(dec)
    action = np.array([dec,order])
    action = np.expand_dims(action,axis=1)
    action = action.T
    next_state,reward,done,_ = env.step(action)
    bal, s_held, s_sold, cost, sales, net, prof = env.render()
        
    replay_buffer.push(state,state_order,act,dec,reward,next_state,done)
    
    frame_idx += 1
    state = next_state
    
    if len(replay_buffer) > replay_initial:
        ord_l, act_l, TD_Loss = compute_td_loss(batch_size)
    
    if (frame_idx%1000)==0:
        weights_act.append(w_act);weights_ord.append(w_ord)
        print('Step-', str(frame_idx), '/', str(num_frames), '| Profit-', prof,'| Model Loss-', ord_l)
        torch.save({'model_state_dict': policy_net_act.state_dict(), 'optimizer_state_dict': optimizer_order.state_dict(), 'loss': TD_Loss},checkpoint_name+'/policy_net_act.pth.tar') #save PolicyNet
        torch.save({'model_state_dict': policy_net_order.state_dict(), 'optimizer_state_dict': optimizer_order.state_dict(), 'loss': TD_Loss},checkpoint_name+'/policy_net_order.pth.tar') #save PolicyNet
        
    rewards.append(reward),loss.append(TD_Loss),loss_actor.append(act_l),loss_order.append(ord_l),profit.append(prof)
    
    if done:
        state = env.reset()

data_save = {}
data_save['weights_act'] = weights_act
data_save['weights_ord'] = weights_ord
data_save['prediction'] = forecast
data_save['action'] = action_list
data_save['reward'] = profit;data_save['loss'] = loss;data_save['act_loss'] = loss_actor;data_save['order_loss'] = loss_order
with open('./data_save.pkl', 'wb') as f: #data+same as frame folder
    pickle.dump(data_save, f)

print('-'*100)
print('Training Completed')



