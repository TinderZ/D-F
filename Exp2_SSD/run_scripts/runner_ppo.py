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
import collections
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


class Runner_ppo:
    def __init__(self, args):
        env, action_num = make_env(args)
        self.env = env
        self.args = args
        self.args.action_num = action_num

    def run(self, num):
        self.agents = [PPO(self.args, self.args.action_num) for _ in range(self.args.num_agents)]
        self.writer = SummaryWriter("/root/autodl-tmp/exp_data/" + self.args.env + "/" + self.args.algorithm + "/" + str(num))
        train_steps = 0
        epi_train_o, epi_train_u, epi_train_u_probability, epi_train_r, epi_train_o_next = [], [], [], [], []
        train_infos = collections.defaultdict(list)
        for epi in tqdm(range(self.args.num_episodes)):
            print('Env {}, Run {}, train episode {}'.format(self.args.env, num, epi))
            epi_o, epi_u, epi_u_probability, epi_r, epi_o_next, epi_terminate = [], [], [], [], [], []
            _, observation = self.env.reset()
            for i in range(self.args.num_agents):
                observation["agent-" + str(i)] = observation["agent-" + str(i)] / 256
            terminated = False
            step = 0
            # epsilon
            epsilon = np.min([1, self.args.epsilon_init + (
                        self.args.epsilon_final - self.args.epsilon_init) * epi / self.args.epsilon_steplen])

            while not terminated and step < self.args.num_steps_train:
                o, u, u_probability, r, o_next, terminate = [], [], [], [], [], []
                actions_dict = {}
                for i in range(self.args.num_agents):
                    o.append(observation["agent-" + str(i)])
                    action, action_logprobability = self.agents[i].choose_action(o[i], epsilon)
                    u.append(action)
                    u_probability.append(action_logprobability)
                    actions_dict["agent-" + str(i)] = action
                _, observation_next, reward, dones, infos = self.env.step(actions_dict)
                for agent_id, info_dict in infos.items():
                    for k, v in info_dict.items():
                        train_infos[f"Train_infos/{agent_id}/{k}"].append(v)

                for i in range(self.args.num_agents):
                    observation_next["agent-" + str(i)] = observation_next["agent-" + str(i)] / 256
                    o_next.append(observation_next["agent-" + str(i)])
                    r.append(reward["agent-" + str(i)])
                    terminate.append(dones["agent-" + str(i)])
                epi_o.append(o)
                epi_u.append(u)
                epi_u_probability.append(u_probability)
                epi_r.append(r)
                epi_o_next.append(o_next)

                observation = observation_next
                step += 1

            epi_train_o.append(epi_o)
            epi_train_u.append(epi_u)
            epi_train_u_probability.append(epi_u_probability)
            epi_train_r.append(epi_r)
            epi_train_o_next.append(epi_o_next)
            if epi % self.args.training_epi_gap == 0:
                episode = dict(o=epi_train_o.copy(),
                               u=epi_train_u.copy(),
                               u_probability=epi_train_u_probability.copy(),
                               r=epi_train_r.copy(),
                               o_next=epi_train_o_next.copy(),
                               )
                epi_train_o, epi_train_u, epi_train_u_probability, epi_train_r, epi_train_o_next = [], [], [], [], []
                train_steps += 1
                for i in range(self.args.num_agents):
                    closs, aloss = self.agents[i].learn(episode, i)
                    self.writer.add_scalar("Agent_{}_CLoss".format(str(i)), closs, train_steps)
                    self.writer.add_scalar("Agent_{}_ALoss".format(str(i)), aloss, train_steps)

                # --- Log training metrics ---
                num_agents = self.args.num_agents
                # The values in train_infos are per-step, so we average them over all steps in the training window
                train_avg_individual_reward = [np.mean(train_infos[f'Train_infos/agent-{i}/reward']) for i in range(num_agents)]
                train_total_reward = sum(train_avg_individual_reward)

                self.writer.add_scalar("Train_Total_reward", train_total_reward, train_steps)
                for i in range(num_agents):
                    self.writer.add_scalar(f"Train_Agent_{i}_reward", train_avg_individual_reward[i], train_steps)

                if self.args.env == "Harvest":
                    train_apples_collected_list = [np.mean(train_infos[f'Train_infos/agent-{i}/apples_collected']) for i in range(num_agents)]
                    variance, std_dev, gini = get_fairness_metrics(train_apples_collected_list)
                    self.writer.add_scalar("Train_Apples_Variance", variance, train_steps)
                    self.writer.add_scalar("Train_Apples_StdDev", std_dev, train_steps)
                    self.writer.add_scalar("Train_Apples_Gini", gini, train_steps)
                    print(f"training episode {epi}, total_reward {train_total_reward:.2f}, individual_rewards {[round(r, 2) for r in train_avg_individual_reward]}, individual_apples{[round(a, 2) for a in train_apples_collected_list]}")
                elif self.args.env == 'Cleanup':
                    train_apples_collected_list = [np.mean(train_infos[f'Train_infos/agent-{i}/apples_collected']) for i in range(num_agents)]
                    train_wastes_cleaned_list = [np.mean(train_infos[f'Train_infos/agent-{i}/wastes_cleaned']) for i in range(num_agents)]
                    variance, std_dev, gini = get_fairness_metrics(train_apples_collected_list)
                    self.writer.add_scalar("Train_Apples_Variance", variance, train_steps)
                    self.writer.add_scalar("Train_Apples_StdDev", std_dev, train_steps)
                    self.writer.add_scalar("Train_Apples_Gini", gini, train_steps)
                    variance, std_dev, gini = get_fairness_metrics(train_wastes_cleaned_list)
                    self.writer.add_scalar("Train_Wastes_Variance", variance, train_steps)
                    self.writer.add_scalar("Train_Wastes_StdDev", std_dev, train_steps)
                    self.writer.add_scalar("Train_Wastes_Gini", gini, train_steps)
                    print(f"training episode {epi}, total_reward {train_total_reward:.2f}, individual_rewards {[round(r, 2) for r in train_avg_individual_reward]}, individual_apples{[round(a, 2) for a in train_apples_collected_list]}")
                
                if self.args.env == "Harvest":
                    train_sustainability_list = [np.mean(train_infos[f'Train_infos/agent-{i}/sustainability']) for i in range(num_agents)]
                    self.writer.add_scalar("Train_Total_sustainability", np.sum(train_sustainability_list), train_steps)
                    for i in range(num_agents):
                        self.writer.add_scalar(f"Train_Agent_{i}_sustainability", train_sustainability_list[i], train_steps)

                train_infos.clear()

                # --- Log evaluation metrics (1 episode) ---
                eval_infos = collections.defaultdict(list)
                _, observation = self.env.reset()
                for i in range(self.args.num_agents):
                    observation["agent-" + str(i)] = observation["agent-" + str(i)] / 256
                for istep in range(self.args.num_steps_evaluate):
                    actions_dict = {}
                    for i in range(self.args.num_agents):
                        action, action_logprobability = self.agents[i].choose_action(observation["agent-" + str(i)], 1)
                        actions_dict["agent-" + str(i)] = action
                    _, observation_next, reward, dones, infos = self.env.step(actions_dict)
                    for agent_id, info_dict in infos.items():
                        for k, v in info_dict.items():
                            eval_infos[f"eval_infos/{agent_id}/{k}"].append(v)
                    for i in range(self.args.num_agents):
                        observation_next["agent-" + str(i)] = observation_next["agent-" + str(i)] / 256
                    observation = observation_next

                # cal social metrics for the single evaluation episode
                # The values in eval_infos are per-step, so we sum them for the whole episode
                eval_individual_reward = [np.sum(eval_infos[f'eval_infos/agent-{i}/reward']) for i in range(num_agents)]
                eval_total_reward = sum(eval_individual_reward)
                self.writer.add_scalar("eval_Total_reward", eval_total_reward, train_steps)
                for i in range(num_agents):
                    self.writer.add_scalar(f"eval_Agent_{i}_reward", eval_individual_reward[i], train_steps)

                if self.args.env == "Harvest":
                    eval_apples_collected_list = [np.sum(eval_infos[f'eval_infos/agent-{i}/apples_collected']) for i in range(num_agents)]
                    variance, std_dev, gini = get_fairness_metrics(eval_apples_collected_list)
                    self.writer.add_scalar("eval_Apples_Variance", variance, train_steps)
                    self.writer.add_scalar("eval_Apples_StdDev", std_dev, train_steps)
                    self.writer.add_scalar("eval_Apples_Gini", gini, train_steps)
                    print(f"evaluating episode {epi}, total_reward {eval_total_reward:.2f}, individual_rewards {[round(r, 2) for r in eval_individual_reward]}, individual_apples{[round(a, 2) for a in eval_apples_collected_list]}")
                elif self.args.env == 'Cleanup':
                    eval_apples_collected_list = [np.sum(eval_infos[f'eval_infos/agent-{i}/apples_collected']) for i in range(num_agents)]
                    eval_wastes_cleaned_list = [np.sum(eval_infos[f'eval_infos/agent-{i}/wastes_cleaned']) for i in range(num_agents)]
                    variance, std_dev, gini = get_fairness_metrics(eval_apples_collected_list)
                    self.writer.add_scalar("eval_Apples_Variance", variance, train_steps)
                    self.writer.add_scalar("eval_Apples_StdDev", std_dev, train_steps)
                    self.writer.add_scalar("eval_Apples_Gini", gini, train_steps)
                    variance, std_dev, gini = get_fairness_metrics(eval_wastes_cleaned_list)
                    self.writer.add_scalar("eval_Wastes_Variance", variance, train_steps)
                    self.writer.add_scalar("eval_Wastes_StdDev", std_dev, train_steps)
                    self.writer.add_scalar("eval_Wastes_Gini", gini, train_steps)
                    print(f"evaluating episode {epi}, total_reward {eval_total_reward:.2f}, individual_rewards {[round(r, 2) for r in eval_individual_reward]}, individual_apples{[round(a, 2) for a in eval_apples_collected_list]}")

            if self.args.env == "Harvest":
                eval_sustainability_list = [np.sum(eval_infos[f'eval_infos/agent-{i}/sustainability']) / self.args.num_steps_evaluate for i in range(num_agents)]
                self.writer.add_scalar("eval_Total_sustainability", np.sum(eval_sustainability_list), train_steps)
                for i in range(num_agents):
                    self.writer.add_scalar(f"eval_Agent_{i}_sustainability", eval_sustainability_list[i], train_steps)
        self.writer.close()












