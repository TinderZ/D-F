import numpy as np
class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.args = args
        print('Init RolloutWorker')

    def generate_episode(self, episode_num, evaluate=False):
        if evaluate:
            max_steps = self.args.num_steps_evaluate
        else:
            max_steps = self.args.num_steps_train
        epi_o, epi_u, epi_u_probability, epi_r, epi_o_next, epi_u_next, epi_u_next_probability, epi_terminate = [], [], [], [], [], [], [], []
        _, observation = self.env.reset()
        for i in range(self.args.num_agents):
            observation["agent-" + str(i)] = observation["agent-" + str(i)] / 256
        terminated = False
        step = 0
        episode_reward = np.zeros(self.args.num_agents)
        episode_apples_collected = np.zeros(self.args.num_agents)
        episode_waste_cleaned = np.zeros(self.args.num_agents)  # 新增: 为waste数创建一个累加器
        episode_sustainability = np.zeros(self.args.num_agents)
        # epsilon
        if evaluate:
            epsilon = 1
        else:
            epsilon = np.min([1, self.args.epsilon_init + (self.args.epsilon_final - self.args.epsilon_init) * episode_num / self.args.epsilon_steplen])

        while not terminated and step < max_steps:
            o, u, u_probability, r, o_next, terminate = [], [], [], [], [], []
            actions_dict = {}
            for i in range(self.args.num_agents):
                o.append(observation["agent-" + str(i)])
                if step == 0:
                    action, action_probability = self.agents[i].choose_action(o[i], epsilon)
                else:
                    action = u_next[i]
                    action_probability = u_next_probability[i]
                u.append(action)
                u_probability.append(action_probability)
                actions_dict["agent-" + str(i)] = action
            _, observation_next, reward, dones, infos = self.env.step(actions_dict)
            u_next, u_next_probability = [], []
            # print(infos)
            for i in range(self.args.num_agents):
                observation_next["agent-" + str(i)] = observation_next["agent-" + str(i)] / 256
                o_next.append(observation_next["agent-" + str(i)])
                r.append(reward["agent-"+str(i)])
                terminate.append(dones["agent-" + str(i)])
                action_next, action_next_probability = self.agents[i].choose_action(o_next[i], epsilon)
                u_next.append(action_next)
                u_next_probability.append(action_next_probability)
                # 从 infos 中累加苹果数量
                agent_key = f"agent-{i}"
                if agent_key in infos and 'apples_collected' in infos[agent_key]:
                    episode_apples_collected[i] += infos[agent_key]['apples_collected']
                # 新增: 如果环境是 'cleanup'，则累加 waste 数量
                if self.args.env == 'Cleanup':
                    if agent_key in infos and 'waste_cleaned' in infos[agent_key]:
                        episode_waste_cleaned[i] += infos[agent_key]['waste_cleaned']
                if self.args.env == 'Harvest':
                    if agent_key in infos and 'sustainability' in infos[agent_key]:
                        episode_sustainability[i] += infos[agent_key]['sustainability']
            episode_reward += np.array(r)
            epi_o.append(o)
            epi_u.append(u)
            epi_u_probability.append(u_probability)
            epi_r.append(r)
            epi_o_next.append(o_next)
            epi_u_next.append(u_next)
            epi_u_next_probability.append(u_next_probability)
            epi_terminate.append(terminate)

            observation = observation_next
            step += 1

        if step > 0 and self.args.env == "Harvest":
            episode_sustainability /= step
        episode = dict(o=epi_o.copy(),
                       u=epi_u.copy(),
                       u_probability_all=epi_u_probability.copy(),
                       r=epi_r.copy(),
                       o_next=epi_o_next.copy(),
                       u_next=epi_u_next.copy(),
                       terminate=epi_terminate.copy()
                       )

        return episode, episode_reward, episode_apples_collected, episode_waste_cleaned, episode_sustainability
