"""
Solution of Open AI gym environment "Cartpole-v0" (https://gym.openai.com/envs/CartPole-v0) using DQN and Pytorch.
This is a modified version of Pytorch DQN tutorial from https://github.com/mahakal001/reinforcement-learning/tree/master/cartpole-dqn 
"""

from dqn_agent import DQN_Agent
import gym
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np

def draw_curve(seed_here, target_update, batch_size):
    env = gym.make('CartPole-v0')
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    exp_replay_size = 256
    agent = DQN_Agent(seed=seed_here, layer_sizes=[input_dim, 64, output_dim], lr=1e-3, sync_freq=target_update,
                    exp_replay_size=exp_replay_size)

    # Main training loop
    losses_list, reward_list, episode_len_list, epsilon_list = [], [], [], []
    episodes = 20000
    epsilon = 1


    # initiliaze experiance replay
    index = 0
    for i in range(exp_replay_size):
        obs = env.reset()
        done = False
        while not done:
            A = agent.get_action(obs, env.action_space.n, epsilon=1)
            obs_next, reward, done, _ = env.step(A.item())
            agent.collect_experience([obs, A.item(), reward, obs_next])
            obs = obs_next
            index += 1
            if index > exp_replay_size:
                break

    index = 128
    for i in tqdm(range(episodes)):
        obs, done, losses, ep_len, rew = env.reset(), False, 0, 0, 0
        while not done:
            ep_len += 1
            A = agent.get_action(obs, env.action_space.n, epsilon)
            obs_next, reward, done, _ = env.step(A.item())
            agent.collect_experience([obs, A.item(), reward, obs_next])

            obs = obs_next
            rew += reward
            index += 1

            if index > 128:
                index = 0
                for j in range(4):
                    loss = agent.train(batch_size)
                    losses += loss
        if epsilon > 0.05:
            epsilon -= (1 / 5000)

        losses_list.append(losses / ep_len), reward_list.append(rew)
        episode_len_list.append(ep_len), epsilon_list.append(epsilon)

    moving_averages= []
    i = 0
    while i < len(reward_list) - 100 + 1:
        this_window = reward_list[i : i + 100]
        window_average = sum(this_window) / 100
        moving_averages.append(window_average)
        i += 1
    
    return moving_averages


target_updates = [1, 10, 30, 100]
seeds = [1423, 1223, 1623]
for i in range(4):
    moving_averages_curve = []
    target_update_here = target_updates[i]
    for j in range(3):
        seed_here = seeds[j]
        moving_averages_curve.append(draw_curve(seed_here, target_update_here, 16))
    mean_here = np.mean(np.array(moving_averages_curve), axis = 0)
    plt.plot(mean_here, label = f"Training Step Size:{target_update_here}")

plt.title('DQN Training for Different Training Step Size')
plt.xlabel('Episode')
plt.ylabel('Running Average Cumulative Reward')
plt.legend()
plt.savefig('./training_step_size_final_ans_18.png')
print("saved")

'''
batch_sizes = [1,16,30,200]
seeds = [1,2,3]
for i in range(4):
    moving_averages_curve = []
    batch_size_here = batch_sizes[i]
    for j in range(3):
        seed_here = seeds[j]
        moving_averages_curve.append(draw_curve(seed_here, 10, batch_size_here))
    mean_here = np.mean(np.array(moving_averages_curve), axis = 0)
    plt.plot(mean_here, label = f"Batch Size:{batch_size_here}")

plt.title('DQN Training for Different Batch Size')
plt.xlabel('Episode')
plt.ylabel('Running Average Cumulative Reward')
plt.legend()
plt.savefig('./batch_size(1,2,3)_time6.png')
print("saved")
'''

# Install the following: 
# Python (3.6 recommended, but should work with later versions as well)
# pip install tqdm 
# conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch (To install pytorch read: https://pytorch.org/)
# pip install 'gym==0.10.11'
# pip install matplotlib


'''
env = gym.make('CartPole-v0')
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
exp_replay_size = 256
target_update = 10
batch_size  = 16
agent = DQN_Agent(seed=1423, layer_sizes=[input_dim, 64, output_dim], lr=1e-3, sync_freq=target_update,
                  exp_replay_size=exp_replay_size)

# Main training loop
losses_list, reward_list, episode_len_list, epsilon_list = [], [], [], []
episodes = 20000
epsilon = 1





# initiliaze experiance replay
index = 0
for i in range(exp_replay_size):
    obs = env.reset()
    done = False
    while not done:
        A = agent.get_action(obs, env.action_space.n, epsilon=1)
        obs_next, reward, done, _ = env.step(A.item())
        agent.collect_experience([obs, A.item(), reward, obs_next])
        obs = obs_next
        index += 1
        if index > exp_replay_size:
            break

index = 128
for i in tqdm(range(episodes)):
    obs, done, losses, ep_len, rew = env.reset(), False, 0, 0, 0
    while not done:
        ep_len += 1
        A = agent.get_action(obs, env.action_space.n, epsilon)
        obs_next, reward, done, _ = env.step(A.item())
        agent.collect_experience([obs, A.item(), reward, obs_next])

        obs = obs_next
        rew += reward
        index += 1

        if index > 128:
            index = 0
            for j in range(4):
                loss = agent.train(batch_size)
                losses += loss
    if epsilon > 0.05:
        epsilon -= (1 / 5000)

    losses_list.append(losses / ep_len), reward_list.append(rew)
    episode_len_list.append(ep_len), epsilon_list.append(epsilon)

plt.figure(2)
plt.clf()
plt.title('Training...')
plt.xlabel('Episode')
plt.ylabel('reward')
moving_averages= []
i = 0
while i < len(reward_list) - 100 + 1:
    this_window = reward_list[i : i + 100]
    window_average = sum(this_window) / 100
    moving_averages.append(window_average)
    i += 1

Ep_arr = np.array(moving_averages)
plt.plot(Ep_arr)
plt.savefig('./cartpole.png')
print("Saving trained model")
agent.save_trained_model("cartpole-dqn.pth")
'''