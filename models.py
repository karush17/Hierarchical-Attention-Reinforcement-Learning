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


class ActorNetwork(nn.Module): 
    def __init__(self, num_inputs, hidden_size):
        super(ActorNetwork, self).__init__()   
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=num_inputs,hidden_size=self.hidden_size,num_layers=1,batch_first=True, bidirectional=True)
        self.attn = ScaledDotAttention(2*self.hidden_size)
        self.fc = nn.Linear(2*self.hidden_size,6)
        
    def forward(self, state): 
        h0 = torch.zeros(2, state.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(2, state.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(state, (h0, c0))
        out,w_act = self.attn(out,out,out)
        out = self.fc(out[:, -1, :])
        return out,w_act

    def act(self, state, epsilon):
        pred,w_act = self.forward(state)
        return pred,w_act

class OrderNetwork(nn.Module): 
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(OrderNetwork, self).__init__()  
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.lstm = nn.LSTM(input_size=num_inputs,hidden_size=self.hidden_size,num_layers=1,batch_first=True, bidirectional=True)
        self.attn = ScaledDotAttention(2*self.hidden_size)
        self.fc = nn.Linear(2*self.hidden_size,self.num_actions)
        
    def forward(self, state): 
        h0 = torch.zeros(2, state.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(2, state.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(state, (h0, c0))
        out,w_ord = self.attn(out,out,out)
        out = self.fc(out[:, -1, :])
        out = F.softmax(out)
        prob = max(out[0,:])
        action = torch.argmax(out,dim=1)
        return out, prob, action,w_ord

    def act(self, state, epsilon):
          if random.random() > epsilon:
            _, prob, action,w_ord = self.forward(state)
          else:
              prob = random.uniform(0,1)
              action = random.randint(0,1)
              w_ord = []
          return prob, action,w_ord


