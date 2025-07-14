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

def calculate_gini(values_list):
    if not isinstance(values_list, list) and not isinstance(values_list, np.ndarray):
        return 0.0
    
    values = np.array(values_list, dtype=np.float64)
    
    if len(values) == 0:
        return 0.0

    mean_val = np.mean(values)

    if mean_val == 0:
        return 0.0

    n = len(values)
    if n <= 1:
        return 0.0
        
    sum_abs_diff = np.sum(np.abs(values - values[:, np.newaxis]))
    denominator = 2 * n**2 * mean_val
    if denominator == 0:
        return 0.0
    return sum_abs_diff / denominator

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
        self.episode_rewards = np.empty([self.args.round, self.args.num_agents, int(self.args.num_episodes/self.args.evaluate_cycle)])
        self.save_data_path = './data/' + self.args.env + str(self.args.num_agents) + '/' + self.args.algorithm


        if not os.path.exists(self.save_data_path):
            os.makedirs(self.save_data_path)

        file = sorted(os.listdir(self.save_data_path))
        if file == []:
            self.next_num = 1
        else:
            self.next_num = int(file[-1].split('.')[0][-1]) + 1

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
        self.writer = SummaryWriter("./runs/" + self.args.env + str(self.args.num_agents) + "/" + self.args.algorithm + "/" + str(num))

        train_steps = 0
        for epi in tqdm(range(self.args.num_episodes)):
            print('Env {}, Run {}, train episode {}'.format(self.args.env, num, epi))

            # The return values of generate_episode are: episode_data, episode_reward, episode_apples_collected, episode_waste_cleaned
            episode_data, episode_reward, episode_apples_collected, _ = self.rolloutWorker.generate_episode(epi)
            train_total_reward = np.sum(episode_reward)
            print(f"training episode {epi} (non-eval), total_reward {train_total_reward}, individual_rewards {episode_reward}, individual_apples{episode_apples_collected}")
            self.buffer.add(episode_data)
            if self.args.batch_size < self.buffer.__len__():
                for train_step in range(self.args.train_steps):
                    if self.args.algorithm == "QMIX":
                        mini_batch = self.buffer.sample(min(self.buffer.__len__(), self.args.batch_size))
                        loss = self.agents.learn(mini_batch)
                        self.writer.add_scalar("Agent_Total_Loss", loss, train_steps)
                    elif self.args.algorithm == "QMIX_SHARE":
                        mini_batch = self.buffer.sample(min(self.buffer.__len__(), self.args.batch_size))
                        loss = self.agents.learn(mini_batch)
                        self.writer.add_scalar("Agent_Total_Loss", loss, train_steps)
                    elif self.args.algorithm == "QMIX_SHARE_STATE":
                        mini_batch = self.buffer.sample(min(self.buffer.__len__(), self.args.batch_size))
                        loss = self.agents.learn(mini_batch)
                        self.writer.add_scalar("Agent_Total_Loss", loss, train_steps)
                    elif self.args.algorithm == "VDN_SHARE":
                        mini_batch = self.buffer.sample(min(self.buffer.__len__(), self.args.batch_size))
                        loss = self.agents.learn(mini_batch)
                        self.writer.add_scalar("Agent_Total_Loss", loss, train_steps)
                    elif self.args.algorithm == "VDN_SHARE_STATE":
                        mini_batch = self.buffer.sample(min(self.buffer.__len__(), self.args.batch_size))
                        loss = self.agents.learn(mini_batch)
                        self.writer.add_scalar("Agent_Total_Loss", loss, train_steps)
                    elif self.args.algorithm == "MADDPG":
                        mini_batch = self.buffer.sample(min(self.buffer.__len__(), self.args.batch_size))
                        for i in range(self.args.num_agents):
                            closs, aloss = self.agents[i].learn(mini_batch, i, self.agents)
                            self.writer.add_scalar("Agent_{}_CLoss".format(str(i)), closs, train_steps)
                            self.writer.add_scalar("Agent_{}_ALoss".format(str(i)), aloss, train_steps)
                    elif self.args.algorithm == "DDPG":
                        for i in range(self.args.num_agents):
                            mini_batch = self.buffer.sample(min(self.buffer.__len__(), self.args.batch_size))
                            closs, aloss = self.agents[i].learn(mini_batch, i)
                            self.writer.add_scalar("Agent_{}_CLoss".format(str(i)), closs, train_steps)
                            self.writer.add_scalar("Agent_{}_ALoss".format(str(i)), aloss, train_steps)
                    elif self.args.algorithm == "DQN" or self.args.algorithm == "DQN-AVG" or self.args.algorithm == "DQN-MIN" or self.args.algorithm == "DQN-RMF" or self.args.algorithm == "DQN-IA" or self.args.algorithm == "SOCIAL":
                        for i in range(self.args.num_agents):
                            mini_batch = self.buffer.sample(min(self.buffer.__len__(), self.args.batch_size))
                            loss = self.agents[i].learn(mini_batch, i)
                            self.writer.add_scalar("Agent_{}_Loss".format(str(i)), loss, train_steps)
                    else:
                        return None
                    train_steps += 1
            
            if epi % self.args.evaluate_cycle == 0:
                avg_individual_reward, avg_apples_collected, avg_wastes_cleaned = self.evaluate()
                self.episode_rewards[num, :, int(epi/self.args.evaluate_cycle)] = avg_individual_reward

                total_reward = np.sum(avg_individual_reward)
                total_apples_collected = np.sum(avg_apples_collected)
                total_wastes_cleaned = np.sum(avg_wastes_cleaned)

                apples_variance = np.var(avg_apples_collected)
                apples_std_dev = np.std(avg_apples_collected)
                apples_gini = calculate_gini(avg_apples_collected)

                wastes_variance = np.var(avg_wastes_cleaned)
                wastes_std_dev = np.std(avg_wastes_cleaned)
                wastes_gini = calculate_gini(avg_wastes_cleaned)

                for i in range(self.args.num_agents):
                    self.writer.add_scalar(f"Agent_{i}_reward", avg_individual_reward[i], epi)
                    self.writer.add_scalar(f"Agent_{i}_apples_collected", avg_apples_collected[i], epi)
                    self.writer.add_scalar(f"Agent_{i}_wastes_cleaned", avg_wastes_cleaned[i], epi)

                self.writer.add_scalar("Total_reward", total_reward, epi)
                self.writer.add_scalar("Total_apples_collected", total_apples_collected, epi)
                self.writer.add_scalar("Total_wastes_cleaned", total_wastes_cleaned, epi)

                self.writer.add_scalar("Apples_Variance", apples_variance, epi)
                self.writer.add_scalar("Apples_StdDev", apples_std_dev, epi)
                self.writer.add_scalar("Apples_Gini", apples_gini, epi)

                self.writer.add_scalar("Wastes_Variance", wastes_variance, epi)
                self.writer.add_scalar("Wastes_StdDev", wastes_std_dev, epi)
                self.writer.add_scalar("Wastes_Gini", wastes_gini, epi)
                print(f"training episode {epi}, total_reward {total_reward}, individual_rewards {avg_individual_reward}, individual_apples{avg_apples_collected}, algorithm {self.args.algorithm}")

            np.save(self.save_data_path + '/epi_total_reward_{}'.format(str(self.next_num)), self.episode_rewards)

    def evaluate(self):
        all_rewards = []
        all_apples = []
        all_wastes = []
        for epi in range(self.args.evaluate_epi):
            # The return values of generate_episode are different for different algorithms.
            # For SOCIAL, it returns: episode, episode_reward, episode_apples_collected, episode_waste_cleaned
            # For others, it might return: episode, episode_reward, win_tag, episode_apples, episode_waste
            # We handle this by checking the length of the returned tuple.
            # According to the check, all rollout workers return 4 values.
            _, episode_reward, episode_apples, episode_waste = self.rolloutWorker.generate_episode(epi, evaluate=True)

            all_rewards.append(episode_reward)
            all_apples.append(episode_apples)
            all_wastes.append(episode_waste)
        
        # 返回每个agent在所有episodes中的平均值
        return np.mean(all_rewards, axis=0), np.mean(all_apples, axis=0), np.mean(all_wastes, axis=0)










