import gym
import math

from types import SimpleNamespace
from cartpole import cartpole_learner, ENV_NAME

def random_policy(observation):
    return env.action_space.sample()

def my_policy(observation):

    def get_state(observation):
        pole_angle = observation[2] + 0.418             # "Normalize" pole angle to number between 0 and 0.836
        state_length = 0.836 / 10                        # Define 5 states of equal length
        state = math.trunc(pole_angle / state_length)   # Calculate in which state we are
        return state

    state = get_state(observation)
    action = 0 if state <= 6 else 1

    print(f"state: {state} -> action: {action}")

    return action

def perfect_policy(obs):
    theta, w = obs[2:4]
    if abs(theta) < 0.03:
        return 0 if w < 0 else 1
    else:
        return 0 if theta < 0 else 1

def play_policy(agent: cartpole_learner, episodes):

    obs = env.reset()[0]

    done = False
    trunc = False
    episode_score = 0

    while not done and not trunc:

        action = agent.policy(obs)
        
        obs, score, done, trunc, _ = env.step(action)
        episode_score += score
        
        env.render()
        reward = -1000 if done and not trunc else episode_score

        agent.learn(obs, reward, score)

        if done or trunc:
            agent.end_episode()
            episodes -= 1
            if episodes != 0:
                done = False
                trunc = False
                episode_score = 0
                env.reset()

env = gym.make(ENV_NAME, render_mode="rgb_array")
agent = cartpole_learner(epsilon=1, alpha=0.2, gamma=0.9, epsilon_change=-0.0005, epsilon_min=0)
print(agent)

play_policy(agent, 3000)
print(agent)
agent.plot_stats()
agent.plot_observations_actions()
env.close()

print(agent.q_table_string())
agent.stop_learning()
env = gym.make(ENV_NAME)
agent.stats.actions = []
agent.stats.observations = []
play_policy(agent, 100)
agent.plot_stats()
agent.plot_observations_actions()
env.close()

# For Video
parameter1 = [
    # 1 That doesn't seem to work :(
    # pole_angle = 11, pole_velocity = 5; reward = score
    SimpleNamespace(epsilon=1, alpha=0.2, gamma=0.5, epsilon_change=-0.001, episodes = 2000),

    # 2 Ah well, we can see an effort
    # pole_angle = 11, pole_velocity = 5; reward = -100 if done and not trunc else score; 
    SimpleNamespace(epsilon=1, alpha=0.2, gamma=0.5, epsilon_change=-0.001, episodes = 2000),

    # 3 Same as before, even states -> worse
    # pole_angle = 10, pole_velocity = 6; reward = -100 if done and not trunc else score; 
    SimpleNamespace(epsilon=1, alpha=0.2, gamma=0.5, epsilon_change=-0.001, episodes = 2000),

    # 4 Double the amount of substates -> better! at least at the start -> high risk high reward? -> increase penalty to lose?
    # pole_angle = 22, pole_velocity = 10; reward = -100 if done and not trunc else score; 
    SimpleNamespace(epsilon=1, alpha=0.2, gamma=0.5, epsilon_change=-0.001, episodes = 2000),

    # 5 Same as before odd states -> better, maybe! still high risk/reward -> also drop after epsilon is min
    # pole_angle = 21, pole_velocity = 9; reward = -100 if done and not trunc else score; 
    SimpleNamespace(epsilon=1, alpha=0.2, gamma=0.5, epsilon_change=-0.001, episodes = 2000),

    # 6 Same as before but 10x more penalty for losing -> about the same /shrug
    # pole_angle = 21, pole_velocity = 9; reward = -1000 if done and not trunc else score; 
    SimpleNamespace(epsilon=1, alpha=0.2, gamma=0.5, epsilon_change=-0.001, episodes = 2000),

    # 7 Different variation of 5 with slower epsilon degrade -> maybe netter, definetely slower and steadier, still so many low values
    # pole_angle = 21, pole_velocity = 9; reward = -100 if done and not trunc else score; 
    SimpleNamespace(epsilon=1, alpha=0.2, gamma=0.5, epsilon_change=-0.0005, episodes = 2000),

    # 8 Combine the higher penalty and the slower epsilon decay -> best so far, still performance decrease around epsilon = 0.3 and many low values
    # pole_angle = 21, pole_velocity = 9; reward = -1000 if done and not trunc else score; 
    SimpleNamespace(epsilon=1, alpha=0.2, gamma=0.5, epsilon_change=-0.0005, episodes = 2000),

    # 9 Crazy idea: reward = episode score -> worse!
    # pole_angle = 21, pole_velocity = 9; reward = -1000 if done and not trunc else episode_score; 
    SimpleNamespace(epsilon=1, alpha=0.2, gamma=0.5, epsilon_change=-0.0005, episodes = 2000),

    # 10 like 8 but keep epsilon at 0.3 min -> das ist wohl eher nd das problem
    # pole_angle = 21, pole_velocity = 9; reward = -1000 if done and not trunc else score; 0.3 EPSILON MIN
    SimpleNamespace(epsilon=1, alpha=0.2, gamma=0.5, epsilon_change=-0.0005, episodes = 2000),

    # 11 like 8 but double the substates (keep ood) -> meh
    # pole_angle = 41, pole_velocity = 17; reward = -1000 if done and not trunc else score; 
    SimpleNamespace(epsilon=1, alpha=0.2, gamma=0.5, epsilon_change=-0.0005, episodes = 2000),

    # 12 Like 8 but higher gamma value -> dunno, compare with 8
    # pole_angle = 21, pole_velocity = 9; reward = -1000 if done and not trunc else score; 
    SimpleNamespace(epsilon=1, alpha=0.2, gamma=0.9, epsilon_change=-0.0005, episodes = 2000),

    # 13 Like 12 but epsilon starts at 0.9 and decays down to 0 (also 300 epsiodes) -> niceo we brought the min up
    # pole_angle = 21, pole_velocity = 9; reward = -1000 if done and not trunc else score; 
    SimpleNamespace(epsilon=0.9, alpha=0.2, gamma=0.9, epsilon_change=-0.0005, epsilon_min=0, episodes = 3000),

    # 14 new epsilon decay strategy !!!!!!!!s
    # pole_angle = 21, pole_velocity = 9; reward = -1000 if done and not trunc else score; 
    SimpleNamespace(epsilon=0.9, alpha=0.2, gamma=0.9, epsilon_change=0.0005, epsilon_min=0, episodes = 3000),
]

parameter2 = [
    # 1
    # pole_angle = 21, pole_velocity = 21; reward = -100 if done and not trunc else score;
    SimpleNamespace(epsilon=1, alpha=0.2, gamma=0.9, epsilon_change=-0.0005, epsilon_min=0, episodes = 3000),

    # 2
    # pole_angle = 21, pole_velocity = 21; reward = -1000 if done and not trunc else score;
    SimpleNamespace(epsilon=1, alpha=0.2, gamma=0.9, epsilon_change=-0.0005, epsilon_min=0, episodes = 3000),

    # 3
    # pole_angle = 21, pole_velocity = 21; reward = -1000 if done and not trunc else episode_score;
    SimpleNamespace(epsilon=1, alpha=0.2, gamma=0.9, epsilon_change=-0.0005, epsilon_min=0, episodes = 3000)
]

# TODO
# See Q-Values as propabilities instead of taking max?
# Config Object (and factory)
# States definition as parameter
# Make playing a method of the agent (parameters: env and episodes)
# export agent
# stop when not learning anymore ? 
# more epsilon degrading startegies
# disconnect (score) stats from agent
# Not sure if moving average is nice