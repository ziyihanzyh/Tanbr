import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .HierTree import HierTree

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Model(nn.Module):

    def __init__(self, input_size, hidden_size, out_size):
        super().__init__()
        self.affine1 = nn.Linear(input_size, hidden_size)
        self.affine2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = self.affine2(x)
        return x * torch.sigmoid(x)

class ReplayBuffer:

    def __init__(self, d,d2, capacity):
        self.buffer = {'context':np.zeros((capacity, d)), 'reward': np.zeros((capacity,d2))}
        self.capacity = capacity
        self.size = 0
        self.pointer = 0


    def add(self, context, reward):
        self.buffer['context'][self.pointer] = context
        self.buffer['reward'][self.pointer] = reward
        self.size = min(self.size+1, self.capacity)
        self.pointer = (self.pointer+1)%self.capacity

    def sample(self, n):
        idx = np.random.randint(0,self.size,size=n)
        return self.buffer['context'][idx], self.buffer['reward'][idx]

class HT_NeuralUCB:
    def __init__(self, d, K, outputsize, beta=10, lamb=1, hidden_size=64, lr=1e-2, reg=0.00001,nu = 1, rho = 1,nucb=False):
        self.nucb = nucb
        self.K = K
        self.T = 0
        self.reg = reg
        self.beta = beta
        self.net = Model(d, hidden_size, outputsize)
        self.hidden_size = hidden_size
        self.nu = nu
        self.rho = rho
        self.net.to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.numel = sum(w.numel() for w in self.net.parameters() if w.requires_grad)
        self.sigma_inv = lamb * np.eye(self.numel, dtype=np.float32)
        self.device = device
        self.outsize = outputsize

        self.theta0 = torch.cat(
                [w.flatten() for w in self.net.parameters() if w.requires_grad]
            )
        self.replay_buffer = ReplayBuffer(d, outputsize,10000)
        self.p_num = 1
        self.get_context = HierTree(nu=nu, rho=rho, domain_dim=d, p_num = self.p_num)
        self.arm = None
        self.arm_con = None


    def take_action(self, weight, evaluate=False):
        if self.nucb:
            context = np.random.dirichlet(np.ones(self.outsize), size=20)
            depth = [1]*20
        else:
            context, depth = self.get_context.get_contexts()
        #

        if self.T<self.K:
            arm = np.random.randint(len(context))
            self.arm = arm
            self.arm_con = context[arm]
            return context[arm]
        # context = context.clone().detach().float()
        context = torch.tensor(context, dtype=torch.float32)
        context = context.to(self.device)
        
        weight = np.array(weight)
        weight = weight[:, None]

        with torch.no_grad():
            tmp1 = - np.dot(self.net(context).cpu().numpy(),weight)
        if evaluate:
            p = tmp1
        else:

            g = np.zeros((len(context), self.numel), dtype=np.float32)
            for k in range(len(context)):
                g[k] = self.grad(context[k]).cpu().numpy()

            with torch.no_grad():
                tmp2 = self.beta * np.sqrt(np.matmul(np.matmul(g[:, None, :], self.sigma_inv), g[:, :, None])[:, 0, :])
            depth = np.array(depth)
            depth = depth[:, None]
            tmp3 = self.nu * (self.rho ** depth)
            p = tmp1 + tmp2 + tmp3

        arm = np.argmax(p)
        self.arm = arm
        self.arm_con = context[arm]
        return context[arm]


    def grad(self, x):
        y = self.net(x)
        self.optimizer.zero_grad()
        y.mean().backward()
        return torch.cat(
                [w.grad.detach().flatten() / np.sqrt(self.hidden_size) for w in self.net.parameters() if w.requires_grad]
            ).to(self.device)

    def update(self, reward):
        arm = self.arm
        context = self.arm_con
        context = torch.tensor(context, dtype=torch.float32)
        context = context.to(self.device)
        self.sherman_morrison_update(self.grad(context).cpu().numpy()[:, None])
        self.replay_buffer.add(context.cpu().numpy(), reward)
        self.T += 1
        self.train()
        if not self.nucb:
            self.get_context.receive_reward(arm//self.p_num, sum(reward)/len(reward))

    def sherman_morrison_update(self, v):
        self.sigma_inv -= (self.sigma_inv @ v @ v.T @ self.sigma_inv) / (1+v.T @ self.sigma_inv @ v)

    def train(self):
        if self.T > self.K and self.T % 1 == 0:
            for _ in range(2):
                x, y = self.replay_buffer.sample(64)
                x = torch.tensor(x, dtype=torch.float32).to(self.device)
                y = torch.tensor(y, dtype=torch.float32).to(self.device).view(-1, self.outsize)
                y_hat = self.net(x)
                loss = F.mse_loss(y_hat, y)
                loss += self.reg * torch.norm(torch.cat(
                [w.flatten() for w in self.net.parameters() if w.requires_grad]) - self.theta0)**2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


