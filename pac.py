import gym
import time

# in terminal, display progress bar to show when done
from tqdm import tqdm

# import Q-Learning Agent
from pac_agent import Agent

'''
    This training code is adapted from the OpenAI Gym tutorial,
    https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/,
    with variable name changes and substantial modifications for better ease of
    use, ability to implement to pacman environment, and understanding of program.
'''

#testing function
def test(agent, env, training_episodes):

    tot_score, tot_time, max_score, max_time = 0,0,0,0
    min_score, min_time = float('inf'), float('inf')

    for episode in tqdm(range(training_episodes)):
        # reset the environment to get the first observation
        done = False
        obs, info = env.reset()
        score = 0

        s = time.time()

        # play one episode (game of pacman)
        while not done:
            
            # will choose best move
            action = int(agent.choose_action(obs, testing=True))

            # gather info and tally score
            next_obs, reward, finished, truncated, info = env.step(action)
            score += reward

            # update current observation and check if the environment is done
            done = finished or truncated
            
            obs = next_obs
        
        # record and compare time and score
        e = time.time()
        elapsed = e-s
        
        if elapsed < min_time:
            min_time = elapsed
        elif elapsed > max_time:
            max_time = elapsed
        
        tot_time += elapsed
        
        if score < min_score:
            min_score = score
        elif score > max_score:
            max_score = score
        
        tot_score += score

    #print results to console
    average_reward = tot_score / training_episodes
    average_time = tot_time / training_episodes
    print("TESTING RESULTS\n________________________")
    print(f'Max Score: {max_score} pts')
    print(f'Min Score: {min_score} pts')
    print(f"Average Score: {average_reward} pts")

    print("Max Survival time: %.3f s" % max_time)
    print("Min Survival time: %.3f s" % min_time)
    print("Average Survival time: %.3f s" % average_time)
    env.close()

def main():
    # create environment and render testing phase
    env = gym.make('ALE/MsPacman-v5')

    # hyperparameters
    learning_rate=0.01  
    discount_factor=0.9
    epsilon_greedy=0.9
    epsilon_min=0.1
    epsilon_decay=0.95
    training_episodes=10
    agent = Agent(
        action_space = env.action_space,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon_greedy=epsilon_greedy,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay
    )

    # keep track of stats for each episode during training, i.e. rewards and lives
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=training_episodes)

    # training loop
    s = time.time()
    for episode in tqdm(range(training_episodes)):
        # reset the environment to get the first observation
        done = False
        obs, info = env.reset()
        
        # detect life lost, get num lives
        prev = info['lives']

        # play one game of MSPACMAN
        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, finished, truncated, info = env.step(action)

            # if life counter decreases, lost life, PUNISH
            if info['lives'] < prev:
                prev=info['lives']
                reward = -1
            
            # update agent
            agent._learn(info, obs, action, reward, finished, next_obs)

            # update current observation and check if the environment is done
            done = finished or truncated
            obs = next_obs

        agent._adjust_epsilon()

    e = time.time()
    elapsed = e-s

    # test the now trained agent
    testing_episodes = 5
    env_test = gym.make('ALE/MsPacman-v5', render_mode='human')
    test(agent, env_test, testing_episodes)

    env.close()

if __name__ == '__main__':
    main()