import argparse
import pickle
from collections import namedtuple
from itertools import count
import os, time
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
import numpy as np
import torch
from models.mappo_actor_critic import MAPPO_Actor, MAPPO_Critic


class MAPPO():
    def __init__(self, args, action_num):
        super(MAPPO, self).__init__()
        self.action_num = action_num
        self.args = args
        self.actor_net = MAPPO_Actor(self.args, self.action_num)
        self.old_actor_net = MAPPO_Actor(self.args, self.action_num)
        self.critic_net = MAPPO_Critic(self.args)
        self.actor_net.load_state_dict(self.old_actor_net.state_dict())
        self.optimizer = optim.Adam([{'params': self.actor_net.parameters(), 'lr': self.args.actor_lr},
                                     {'params': self.critic_net.parameters(), 'lr': self.args.critic_lr}])
        if self.args.cuda:
            self.actor_net.cuda()
            self.old_actor_net.cuda()
            self.critic_net.cuda()

    def choose_action(self, obs, episolon, agent_id):
        obsnp = np.array(obs)
        obs = torch.from_numpy(obsnp.copy()).unsqueeze(dim=0)
        obs = obs.float()
        if self.args.cuda:
            obs = obs.cuda()
        with torch.no_grad():
            action_prob = self.old_actor_net(obs)
            dist = Categorical(action_prob)
            if np.random.rand() <= episolon:
                action = dist.sample()
                action_logprob = dist.log_prob(action)
            else:
                action = torch.randint(0, self.action_num, (1,1))
                if self.args.cuda:
                    action = action.cuda()
                action_logprob = dist.log_prob(action)
        action = action.squeeze().cpu().data.numpy()
        action_logprob = action_logprob.squeeze().squeeze().cpu().data.numpy()
        return int(action), action_logprob

    def learn(self, episode_data):
        batch_observation = torch.from_numpy(np.array(episode_data['o'])).float()
        batch_state = torch.from_numpy(np.array(episode_data['s'])).float()
        batch_action = torch.from_numpy(np.array(episode_data['u'])).long()
        batch_action_prob = torch.from_numpy(np.array(episode_data['u_probability'])).float()
        batch_reward = torch.from_numpy(np.array(episode_data['r'])).float()
        batch_state_next = torch.from_numpy(np.array(episode_data['s_next'])).float()
        if self.args.cuda:
            batch_observation = batch_observation.cuda()
            batch_state = batch_state.cuda()
            batch_action = batch_action.cuda()
            batch_action_prob = batch_action_prob.cuda()
            batch_reward = batch_reward.cuda()
            batch_state = batch_state.cuda()
            batch_state_next = batch_state_next.cuda()

        value_loss = 0
        action_loss = 0
        batch_return = torch.zeros(batch_state.shape[0], batch_state.shape[1], 1)
        if self.args.cuda:
            batch_return = batch_return.cuda()
        for step in range(self.args.num_steps_train - 1, -1, -1):
            if step == self.args.num_steps_train - 1:
                batch_return[:, step, ...] = batch_reward[:, step, 0, ...].unsqueeze(-1)
            if step < self.args.num_steps_train - 1:
                batch_return[:, step, ...] = batch_reward[:, step, 0, ...].unsqueeze(-1) + self.args.gamma * batch_return[:, step + 1, ...]
        batch_return = (batch_return - batch_return.mean()) / (batch_return.std() + 1e-5)
        V = self.critic_net(batch_state)
        advantage = (batch_return - V).detach()
        total_value_loss = 0
        total_actor_loss = 0
        for _ in range(self.args.training_times):
            V = self.critic_net(batch_state)
            value_loss = F.mse_loss(batch_return, V)
            actor_loss = 0
            for agent_id in range(self.args.num_agents):
                action_probs = self.actor_net(batch_observation[:, :, agent_id, ...]) # new policy
                action_probs = action_probs.reshape(-1, self.action_num)
                dist = Categorical(action_probs)
                action_prob = dist.log_prob(batch_action[:, :, agent_id, ...].reshape(-1))
                action_prob = action_prob.reshape(self.args.num_steps_train, -1).squeeze(-1)

                ratio = torch.exp(action_prob - batch_action_prob[:, :, agent_id, ...]).unsqueeze(dim=-1)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.args.clip_param, 1 + self.args.clip_param) * advantage
                actor_loss += -torch.min(surr1, surr2).mean()
            loss = actor_loss + value_loss
            total_value_loss += value_loss.item()
            total_actor_loss += actor_loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.args.grad_clip)
            nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.args.grad_clip)
            self.optimizer.step()

        self.old_actor_net.load_state_dict(self.actor_net.state_dict())
        return total_value_loss, total_actor_loss
