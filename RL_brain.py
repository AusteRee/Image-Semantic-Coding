import numpy as np
# import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import torch.optim as optim
from os.path import join
import time
# import copy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# define the network architecture
class Net(nn.Module):

    def __init__(self, Q, a=0.25, in_ch=3, out_ch=16, w=30, h=62):
        super(Net, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.w = w
        self.h = h
        self.al = a
        self.conv1 = nn.Conv2d(in_channels=self.in_ch, out_channels=16, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(16, self.out_ch, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self.out_ch * self.w * self.h + 20, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, Q)
        self.dropout = nn.Dropout2d(p=0.5)
        self.dropout1 = nn.Dropout(p=0.5)
        self.device = torch.device('cuda')
        self.leaky_relu = nn.LeakyReLU(negative_slope=self.al)


    def forward(self, x, m, n, a=0.25):
        onehotM = torch.zeros(19, dtype=torch.int8).to(self.device)
        if m.item() < 19:
            onehotM[m] = 1
        x = F.leaky_relu(self.conv1(x), negative_slope=a)
        x = self.maxpool(x)

        x = F.leaky_relu(self.conv2(x), negative_slope=a)
        x = self.maxpool(x)

        x = torch.flatten(x)
        x = torch.cat([x, onehotM], 0)
        x = torch.cat([x, n], 0)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=0)

        return x


class DeepQNetwork:
    def __init__(self, n_actions, n_features=5, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=25, memory_size=10000, batch_size=1, e_greedy_increment=None,
                 alpha = 0.25):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.beta1 = 0.0
        self.beta2 = 0.999
        self.alpha = alpha
        self.device = torch.device('cuda')

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = torch.zeros((self.memory_size, 3 * 2 + 2))
        self.memory = []

        self.loss_func = nn.MSELoss()
        self.cost_his = []
        self.reward_his = []

        self._build_net()


    def _build_net(self):
        # self.q_eval = Net(self.n_actions).to(self.device)
        # self.q_target = Net(self.n_actions).to(self.device)
        self.q_eval = Net(self.n_features, self.alpha).to(self.device)
        self.q_target = Net(self.n_features, self.alpha).to(self.device)
        self.optimizer = optim.Adam(self.q_eval.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

    def store_transition(self, s, a, r, s_):  # 记忆存储
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # transition = np.hstack((s, [a, r], s_))
        transition = [s, a, r, s_]
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        # self.memory[index, :] = transition
        if self.memory_counter < self.memory_size:
            self.memory.append(transition)
        else:
            self.memory[index] = transition
        self.memory_counter += 1

    def choose_action(self, observation):   #action = RL.choose_action(observation)
                            # observation = [sm, fm, m]
                             #	list {tensor[3, 256, 512], tensor[3, 256, 512], int(m) }
        fm = observation[1].unsqueeze(0).to(self.device)
        m = torch.tensor([observation[2]]).to(self.device)
        n = torch.tensor([observation[0].sum()/(3*256*512)]).to(self.device)
        # observation = torch.Tensor(observation[np.newaxis, :])
        if np.random.uniform() < self.epsilon:
            actions_value = self.q_eval(fm, m, n, self.alpha)

            action = torch.argmax(actions_value, dim=0) #tensor([2], device='cuda:0')
            # action = np.argmax(actions_value.data.numpy())
        else:
            action = torch.randint(0, self.n_actions, (1,)).to(self.device)
            # action = np.random.randint(0, self.n_actions)
        return action     #tensor([2], device='cuda:0')

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())
            print("\ntarget params replaced\n")

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index[0]]
        fm = batch_memory[0][1].unsqueeze(0).to(self.device)
        m = torch.tensor([batch_memory[0][2]]).to(self.device)
        n = torch.tensor([batch_memory[0][0].sum()/(3*256*512)]).to(self.device)
        fm1 = batch_memory[3][1].unsqueeze(0).to(self.device)
        m1 = torch.tensor([batch_memory[3][2]]).to(self.device)
        n1 = torch.tensor([batch_memory[3][0].sum()/(3*256*512)]).to(self.device)

        # q_next is used for getting which action would be choosed by target network in state s_(t+1)
        q_next, q_eval = self.q_target(fm1, m1, n1), self.q_eval(fm, m, n)
        # used for calculating y, we need to copy for q_eval because this operation could keep the Q_value that has not been selected unchanged,
        # so when we do q_target - q_eval, these Q_value become zero and wouldn't affect the calculation of the loss
        q_target = torch.Tensor(q_eval.data.cpu().numpy().copy())

            # q_next, q_eval = self.q_target(torch.Tensor(batch_memory[:, -self.n_features:])), self.q_eval(
            #     torch.Tensor(batch_memory[:, :self.n_features]))
            # used for calculating y, we need to copy for q_eval because this operation could keep the Q_value that has not been selected unchanged,
            # so when we do q_target - q_eval, these Q_value become zero and wouldn't affect the calculation of the loss
            # q_target = torch.Tensor(q_eval.data.numpy().copy())
            # batch_index = np.arange(self.batch_size, dtype=np.int32)
            # eval_act_index = batch_memory[:, self.n_features].astype(int)
            # reward = torch.Tensor(batch_memory[:, self.n_features + 1])
        eval_act_index = batch_memory[1].cpu().numpy()
        reward = batch_memory[2]
        if m < 18:
            q_target[eval_act_index] = reward + self.gamma * torch.max(q_next)
        else:
            q_target[eval_act_index] = reward
        q_target = q_target.to(self.device)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # increase epsilon
        self.cost_his.append(loss)
        self.reward_his.append([reward, m])
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his.cpu())), self.cost_his.cpu())
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


# if __name__ == '__main__':
#     x = torch.randn(1, 3, 256, 512)
#     model = RL_network(5)
#     m = torch.tensor([3])
#     x, Q = model(x, m)
#     print(out)