import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict
from matplotlib.patches import Patch

# import Q-Learning Agent
from blackjack_agent import Q_Agent

'''
    This training code is adapted from the OpenAI Gym tutorial, 
    https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/,
    with some variable name changes and a slight modifications for better ease of
    use, ability to test the trained agent, and understanding of what is occurring.
'''

# visualize the policy learned from training
def create_grids(agent, usable_ace):
    # create value and policy grids for agent
    state_value = defaultdict(float)
    policy = defaultdict(int)

    for obs, action_values in agent.q_table.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    # player's card sum and dealer's face up card
    player_count, dealer_count = np.meshgrid(
        np.arange(12, 22),
        np.arange(1, 11),
    )

    # create the value grid for plotting
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )

    # value = expected return that the agent can achieve starting from that state
    #       and following the current policy
    value_grid = player_count, dealer_count, value

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )

    return value_grid, policy_grid



def create_plots(value_grid, policy_grid):
    # create a new figure with 2 subplots (left: state values, right: policy)
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))

    # plot the state values
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap='plasma',
        edgecolor='none',
    )

    # lighter the color = higher expected total reward based on the combo of
    #   player card sum and dealer's face up card
    plt.xticks(range(12, 22), range(12, 22))
    plt.yticks(range(1, 11), ['A'] + list(range(2, 11)))
    ax1.set_xlabel('Player Card Sum')
    ax1.set_ylabel('Dealer\'s Face Up Card')
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel('Expected Reward Total', fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    # plot the agent's policy
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap='Accent_r', cbar=False)
    ax2.set_xlabel('Player Card Sum')
    ax2.set_ylabel('Dealer\'s Face Up Card')
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(['A'] + list(range(2, 11)), fontsize=12)

    # create legend
    legend_elements = [
        Patch(facecolor='lightgreen', edgecolor='black', label='Hit'),
        Patch(facecolor='grey', edgecolor='black', label='Stick')
    ]

    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))

    return fig



# testing function
def test(agent, env, n_episodes):
    total_rewards = 0
    num_wins = 0

    for episode in range(n_episodes):
        # reset the environment to get the first observation
        done = False
        obs, info = env.reset()

        # play one episode
        while not done:
            # always select the action with the highest Q-value
            action = int(np.argmax(agent.q_table[obs]))
            next_obs, reward, finished, truncated, info = env.step(action)
            total_rewards += reward

            # update win count
            if reward == 1:
                num_wins += 1

            # update current observation and check if the environment is done
            done = finished or truncated
            obs = next_obs


    average_reward = total_rewards / n_episodes
    win_rate = num_wins / n_episodes

    print('Testing Results:')
    print('\tAverage Reward per Episode:', average_reward)
    print('\tWin Rate:', win_rate)



def main():
    # create environment
    env = gym.make('Blackjack-v1', sab=True)

    # episode amounts for training and testing
    n_episodes = 100_000
    testing_episodes = 50_000
    
    # hyperparameters
    learning_rate = 0.01
    discount_factor = 0.95
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1
    
    # determing if and Ace can be both 1 and 11 or just 1
    usable_ace = False

    # initialize q-learning agent
    agent = Q_Agent (
        action_space=env.action_space,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        initial_e=start_epsilon,
        decay_e=epsilon_decay,
        final_e=final_epsilon
    )

    # keep track of stats for each episode during training, i.e. rewards and lengths
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

    train_start = time.time()

    # training loop
    for episode in range(n_episodes):
        # reset the environment to get the first observation
        done = False
        obs, info = env.reset()

        # play one episode
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, finished, truncated, info = env.step(action)

            # update agent
            agent.update(obs, action, reward, finished, next_obs)

            # update current observation and check if the environment is done
            done = finished or truncated
            obs = next_obs
        
        agent.epsilon_decay()

    train_end = time.time()

    # display state values & policy with ace = 1
    value_grid, policy_grid = create_grids(agent, usable_ace)
    train_fig = create_plots(value_grid, policy_grid)
    plt.suptitle('Training Data', fontsize=16)
    plt.show()

    # train_fig.savefig('training_data.png')  
    
    # test the now trained agent
    test_start = time.time()
    test(agent, env, testing_episodes)
    test_end = time.time()

    # print runtimes
    print(f'\nTraining Runtime: {train_end-train_start:.4f} seconds')
    print(f'Testing Runtime: {test_end-test_start:.4f} seconds')

    # close environment
    env.close()

if __name__ == '__main__':
    main()