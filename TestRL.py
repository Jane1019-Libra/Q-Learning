# Code for TestRL.py
import numpy as np
import MDP
import RL
import matplotlib.pyplot as plt


''' Construct simple MDP as described in Lecture 16 Slides 21-22'''
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
R = np.array([[0,0,10,10],[0,0,10,10]])
discount = 0.9
mdp = MDP.MDP(T,R,discount)
rlProblem = RL.RL(mdp,np.random.normal)

# Test Q-learning
'''
[Q,policy] = rlProblem.qLearning(s0=0,initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=1000,nSteps=100,epsilon=0.3)
print("\nQ-learning results")
print(Q)
print(policy)
'''

nTrials = 100
epsilons = [0.05, 0.1, 0.3, 0.5]

for i in range(4):
  average_rewards = np.zeros(200)
  for trials in range(nTrials):
    epsilon_here = epsilons[i]
    [Q,policy, cumulated_reward] = rlProblem.qLearning(s0=0,initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=200,nSteps=100,epsilon=epsilon_here)
    if trials in [0,49,99]:
      print("Epsilon here is: ", epsilon_here)
      print("Q values is: ", Q)
      print("Policy is: ", policy)
      print("")
    average_rewards = average_rewards + cumulated_reward
  average_rewards = average_rewards / nTrials
  plt.plot(np.arange(200), average_rewards, label=f"Epsilon={epsilon_here}")

plt.xlabel("Episodes")
plt.ylabel("Average Cumulative Discounted Reward")
plt.legend()
plt.show()