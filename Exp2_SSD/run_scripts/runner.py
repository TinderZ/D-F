import sys
sys.path.append("../")
import numpy as np
import os
from replay_buffer.replay_buffer_episode import ReplayBuffer
from social_dilemmas.envs.cleanup import CleanupEnv
from social_dilemmas.envs.harvest import HarvestEnv
from ray.tune.registry import register_env
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from learners.DQN import DQN
from learners.SOCIAL import DQN_SOCIAL
from learners.PPO import PPO
from learners.DDPG import DDPG
from learners.MADDPG import MADDPG
from learners.QMIX_SHARE import QMIX_SHARE
from learners.MAPPO import MAPPO
from tqdm import tqdm
from utility_funcs import get_fairness_metrics

def make_env(args):
    if args.env == "Harvest":
        single_env = HarvestEnv(num_agents=args.num_agents)
        env_name = "HarvestEnv"
        def env_creator(_):
            return HarvestEnv(num_agents=args.num_agents)
    elif args.env == "Cleanup":
        single_env = CleanupEnv(num_agents=args.num_agents)
        env_name = "CleanupEnv"
        def env_creator(_):
            return CleanupEnv(num_agents=args.num_agents)
    else:
        return 0
    register_env(env_name, env_creator)
    if env_name == "HarvestEnv":
        action_num = 8
    else:
        action_num = 9
    return single_env, action_num


class Runner:
    def __init__(self, args):
        env, action_num = make_env(args)
        self.env = env
        self.args = args
        self.args.action_num = action_num
        # self.episode_rewards = np.empty([self.args.round, self.args.num_agents, int(self.args.num_episodes/self.args.evaluate_cycle)])

    def run(self, num):
        self.buffer = ReplayBuffer(self.args)
        if self.args.algorithm == "DQN" or self.args.algorithm == "DQN-AVG" or self.args.algorithm == "DQN-MIN" or self.args.algorithm == "DQN-RMF" or self.args.algorithm == "DQN-IA":
            self.agents = [DQN(self.args, self.args.action_num) for _ in range(self.args.num_agents)]
            from run_scripts.rollout import RolloutWorker
        elif self.args.algorithm == "SOCIAL":
            self.agents = [DQN_SOCIAL(self.args, self.args.action_num) for _ in range(self.args.num_agents)]
            from run_scripts.rollout_social import RolloutWorker
        elif self.args.algorithm == "DDPG":
            self.agents = [DDPG(self.args, self.args.action_num) for _ in range(self.args.num_agents)]
            from run_scripts.rollout import RolloutWorker
        elif self.args.algorithm == "MADDPG":
            self.agents = [MADDPG(self.args, self.args.action_num) for _ in range(self.args.num_agents)]
            from run_scripts.rollout_maddpg import RolloutWorker
        elif self.args.algorithm == "QMIX":
            self.agents = QMIX_SHARE(self.args, self.args.action_num)
            from run_scripts.rollout_qmix import RolloutWorker
        else:
            return None

        self.rolloutWorker = RolloutWorker(self.env, self.agents, self.args)
        #self.writer = SummaryWriter("~/tf-logs/" + self.args.env + str(self.args.num_agents) + "/" + self.args.algorithm + "/" + str(num))
        self.writer = SummaryWriter("/root/autodl-tmp/exp_data/" + self.args.env + "/" + self.args.algorithm + "/" + str(num))

        train_steps = 0
        cycle_rewards = []
        cycle_apples = []
        cycle_wastes = [] if self.args.env == 'Cleanup' else None

        for epi in tqdm(range(self.args.num_episodes)):
            print('Env {}, Run {}, train episode {}'.format(self.args.env, num, epi))

            # The return values of generate_episode are: episode_data, episode_reward, episode_apples_collected, episode_waste_cleaned
            rollout_returns = self.rolloutWorker.generate_episode(epi)
            episode_data, episode_reward, episode_apples_collected = rollout_returns[0], rollout_returns[1], rollout_returns[2]
            if self.args.env == 'Cleanup':
                episode_wastes_cleaned = rollout_returns[3]
                cycle_wastes.append(episode_wastes_cleaned)


            cycle_rewards.append(episode_reward)
            cycle_apples.append(episode_apples_collected)
            
            train_total_reward = np.sum(episode_reward)
            print(f"training episode {epi} (non-eval), total_reward {train_total_reward}, individual_rewards {episode_reward}, individual_apples{episode_apples_collected}")
            self.buffer.add(episode_data)
            if self.args.batch_size < self.buffer.__len__():
                for train_step in range(self.args.train_steps):
                    if self.args.algorithm == "QMIX":
                        mini_batch = self.buffer.sample(min(self.buffer.__len__(), self.args.batch_size))
                        loss = self.agents.learn(mini_batch)
                        self.writer.add_scalar("Agent_Total_Loss", loss.item(), train_steps)
                    elif self.args.algorithm == "QMIX_SHARE":
                        mini_batch = self.buffer.sample(min(self.buffer.__len__(), self.args.batch_size))
                        loss = self.agents.learn(mini_batch)
                        self.writer.add_scalar("Agent_Total_Loss", loss.item(), train_steps)
                    elif self.args.algorithm == "QMIX_SHARE_STATE":
                        mini_batch = self.buffer.sample(min(self.buffer.__len__(), self.args.batch_size))
                        loss = self.agents.learn(mini_batch)
                        self.writer.add_scalar("Agent_Total_Loss", loss.item(), train_steps)
                    elif self.args.algorithm == "VDN_SHARE":
                        mini_batch = self.buffer.sample(min(self.buffer.__len__(), self.args.batch_size))
                        loss = self.agents.learn(mini_batch)
                        self.writer.add_scalar("Agent_Total_Loss", loss.item(), train_steps)
                    elif self.args.algorithm == "VDN_SHARE_STATE":
                        mini_batch = self.buffer.sample(min(self.buffer.__len__(), self.args.batch_size))
                        loss = self.agents.learn(mini_batch)
                        self.writer.add_scalar("Agent_Total_Loss", loss.item(), train_steps)
                    elif self.args.algorithm == "MADDPG":
                        mini_batch = self.buffer.sample(min(self.buffer.__len__(), self.args.batch_size))
                        for i in range(self.args.num_agents):
                            closs, aloss = self.agents[i].learn(mini_batch, i, self.agents)
                            self.writer.add_scalar("Agent_{}_CLoss".format(str(i)), closs.item(), train_steps)
                            self.writer.add_scalar("Agent_{}_ALoss".format(str(i)), aloss.item(), train_steps)
                    elif self.args.algorithm == "DDPG":
                        for i in range(self.args.num_agents):
                            mini_batch = self.buffer.sample(min(self.buffer.__len__(), self.args.batch_size))
                            closs, aloss = self.agents[i].learn(mini_batch, i)
                            self.writer.add_scalar("Agent_{}_CLoss".format(str(i)), closs.item(), train_steps)
                            self.writer.add_scalar("Agent_{}_ALoss".format(str(i)), aloss.item(), train_steps)
                    elif self.args.algorithm == "DQN" or self.args.algorithm == "DQN-AVG" or self.args.algorithm == "DQN-MIN" or self.args.algorithm == "DQN-RMF" or self.args.algorithm == "DQN-IA" or self.args.algorithm == "SOCIAL":
                        for i in range(self.args.num_agents):
                            mini_batch = self.buffer.sample(min(self.buffer.__len__(), self.args.batch_size))
                            loss = self.agents[i].learn(mini_batch, i)
                            self.writer.add_scalar("Agent_{}_Loss".format(str(i)), loss.item(), train_steps)
                    else:
                        return None
                    train_steps += 1
            
            if (epi + 1) % self.args.evaluate_cycle == 0:
                avg_individual_reward = np.mean(cycle_rewards, axis=0)
                avg_apples_collected = np.mean(cycle_apples, axis=0)
                if self.args.env == 'Cleanup':
                    avg_wastes_cleaned = np.mean(cycle_wastes, axis=0)
                else:
                    avg_wastes_cleaned = np.zeros(self.args.num_agents)
                # self.episode_rewards[num, :, int(epi/self.args.evaluate_cycle)] = avg_individual_reward

                total_reward = np.sum(avg_individual_reward)
                total_apples_collected = np.sum(avg_apples_collected)
                if self.args.env == 'Cleanup':
                    total_wastes_cleaned = np.sum(avg_wastes_cleaned)

                apples_variance, apples_std_dev, apples_gini = get_fairness_metrics(avg_apples_collected)

                if self.args.env == 'Cleanup':
                    wastes_variance, wastes_std_dev, wastes_gini = get_fairness_metrics(avg_wastes_cleaned)

                for i in range(self.args.num_agents):
                    self.writer.add_scalar(f"Agent_{i}_reward", avg_individual_reward[i], epi)
                    self.writer.add_scalar(f"Agent_{i}_apples_collected", avg_apples_collected[i], epi)
                    if self.args.env == 'Cleanup':
                        self.writer.add_scalar(f"Agent_{i}_wastes_cleaned", avg_wastes_cleaned[i], epi)

                self.writer.add_scalar("Total_reward", total_reward, epi)
                self.writer.add_scalar("Total_apples_collected", total_apples_collected, epi)
                if self.args.env == 'Cleanup':
                    self.writer.add_scalar("Total_wastes_cleaned", total_wastes_cleaned, epi)

                self.writer.add_scalar("Apples_Variance", apples_variance, epi)
                self.writer.add_scalar("Apples_StdDev", apples_std_dev, epi)
                self.writer.add_scalar("Apples_Gini", apples_gini, epi)

                if self.args.env == 'Cleanup':
                    self.writer.add_scalar("Wastes_Variance", wastes_variance, epi)
                    self.writer.add_scalar("Wastes_StdDev", wastes_std_dev, epi)
                    self.writer.add_scalar("Wastes_Gini", wastes_gini, epi)
                print(f"training episode {epi}, total_reward {total_reward}, individual_rewards {avg_individual_reward}, individual_apples{avg_apples_collected}, algorithm {self.args.algorithm}")
                
                # Now perform evaluation and log with 'eval_' prefix
                eval_avg_individual_reward, eval_avg_apples_collected, eval_avg_wastes_cleaned = self.evaluate()
                eval_total_reward = np.sum(eval_avg_individual_reward)
                eval_total_apples_collected = np.sum(eval_avg_apples_collected)
                if self.args.env == 'Cleanup':
                    eval_total_wastes_cleaned = np.sum(eval_avg_wastes_cleaned)

                eval_apples_variance, eval_apples_std_dev, eval_apples_gini = get_fairness_metrics(eval_avg_apples_collected)

                if self.args.env == 'Cleanup':
                    eval_wastes_variance, eval_wastes_std_dev, eval_wastes_gini = get_fairness_metrics(eval_avg_wastes_cleaned)

                for i in range(self.args.num_agents):
                    self.writer.add_scalar(f"eval_Agent_{i}_reward", eval_avg_individual_reward[i], epi)
                    self.writer.add_scalar(f"eval_Agent_{i}_apples_collected", eval_avg_apples_collected[i], epi)
                    if self.args.env == 'Cleanup':
                        self.writer.add_scalar(f"eval_Agent_{i}_wastes_cleaned", eval_avg_wastes_cleaned[i], epi)

                self.writer.add_scalar("eval_Total_reward", eval_total_reward, epi)
                self.writer.add_scalar("eval_Total_apples_collected", eval_total_apples_collected, epi)
                if self.args.env == 'Cleanup':
                    self.writer.add_scalar("eval_Total_wastes_cleaned", eval_total_wastes_cleaned, epi)

                self.writer.add_scalar("eval_Apples_Variance", eval_apples_variance, epi)
                self.writer.add_scalar("eval_Apples_StdDev", eval_apples_std_dev, epi)
                self.writer.add_scalar("eval_Apples_Gini", eval_apples_gini, epi)

                if self.args.env == 'Cleanup':
                    self.writer.add_scalar("eval_Wastes_Variance", eval_wastes_variance, epi)
                    self.writer.add_scalar("eval_Wastes_StdDev", eval_wastes_std_dev, epi)
                    self.writer.add_scalar("eval_Wastes_Gini", eval_wastes_gini, epi)
                
                print(f"training episode {epi}, total_reward {total_reward}, individual_rewards {avg_individual_reward}, individual_apples{avg_apples_collected}, algorithm {self.args.algorithm}")
                
                
                cycle_rewards = []
                cycle_apples = []
                if self.args.env == 'Cleanup':
                    cycle_wastes = []

            

    def evaluate(self):
        all_rewards = []
        all_apples = []
        all_wastes = []
        for epi in range(self.args.evaluate_epi):
            rollout_returns = self.rolloutWorker.generate_episode(epi, evaluate=True)
            episode_reward, episode_apples = rollout_returns[1], rollout_returns[2]
            all_rewards.append(episode_reward)
            all_apples.append(episode_apples)
            if self.args.env == 'Cleanup':
                episode_waste = rollout_returns[3]
                all_wastes.append(episode_waste)

        avg_individual_reward = np.mean(all_rewards, axis=0)
        avg_apples_collected = np.mean(all_apples, axis=0)
        if self.args.env == 'Cleanup':
            avg_wastes_cleaned = np.mean(all_wastes, axis=0)
        else:
            avg_wastes_cleaned = np.zeros(self.args.num_agents)
        return avg_individual_reward, avg_apples_collected, avg_wastes_cleaned










