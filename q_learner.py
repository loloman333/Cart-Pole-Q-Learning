import numpy as np
import math
import matplotlib.pyplot as plt

from types import SimpleNamespace

def moving_avg(array, windows_size):
    windows_size = round(windows_size)
    result = [np.nan for _ in range(0, round(windows_size/2))]
    moving_sum = sum(array[:windows_size])
    result.append(moving_sum / windows_size)
    for i in range(windows_size, len(array)):
        moving_sum += (array[i] - array[i - windows_size])
        result.append(moving_sum / windows_size)
    return result

class q_learner:

    def __init__(self, num_states: int, num_actions: int, epsilon: float, alpha: float, gamma: float, epsilon_change: float, epsilon_min) -> None:

        self.num_states = num_states    
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.epsilon_change = epsilon_change
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        self.gamma = gamma

        self.keep_learning = True
        self.last_state = None
        self.last_action = None
        self.score = 0
        
        self.stats = SimpleNamespace()
        self.stats.scores = []
        self.stats.best_score = 0
        self.stats.epsilon_values = []
        self.stats.stop_learning = []

        self.q_table = np.zeros((num_states, num_actions))
        
    def __str__(self) -> str:
        return f"""
            Q-Learner

            Number of States: {self.num_states}
            Number of Actions: {self.num_actions}
            Q-Table Entries: {self.num_actions * self.num_states}

            Alpha: {self.alpha}
            Gamma: {self.gamma}
            (Current) Epsilon: {self.epsilon}
            Epsilon Change per Episode: {self.epsilon_change}
            Min Epsilon: {self.epsilon_min}

            Number of Episodes in stats: {len(self.stats.scores)}
            Total Average Score: {np.average(self.stats.scores)}
            Still Learning: {self.keep_learning}
        """

    def __repr__(self) -> str:
        return self.__str__()

    def q_table_string(self):
        string = f"  + {'     '.join([f'{i:3}' for i in range(0, self.num_actions)])}\n"

        for i in range(0, self.num_states):
            string += f"{i:3}  "
            for j in range(0, self.num_actions):
                string += f"{self.q_table[i][j]:6.3f}  "
            string += "\n"

        return string.rstrip("\n")

    def stop_learning(self):
        self.keep_learning = False
        self.stats.stop_learning.append(len(self.stats.scores))

    def get_substate(self, value, state_count, state_length):
        negative = True if value < 0 else False
        value = -value if negative else value

        substate = value / state_length
        substate = round(substate) if state_count % 2 != 0 else math.trunc(substate)
        substate = state_count - 1 if substate >= state_count else substate
        substate = -substate if negative else substate
        substate += math.trunc(state_count / 2)

        assert (substate >= 0 and substate < state_count), f"{substate}, {state_count}"

        return substate

    def combine_substates(self, substates, substate_counts):
        state = 0
        mulitplier = 1
        for index, substate in enumerate(substates):
            state += substate * mulitplier
            mulitplier *= substate_counts[index]

        return state

    def get_state(self, observation):
        raise NotImplementedError

    def epsilon_deacy(self):
        #"""
        if self.epsilon > self.epsilon_min: 
            self.epsilon += self.epsilon_change
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min
        """
        if (len(self.stats.scores) > 100):

            if self.score >= np.average(self.stats.scores[-100:-1]):
                self.epsilon -= self.epsilon_change
            else:
                self.epsilon += self.epsilon_change
        """
    def end_episode(self):

        if self.score > self.stats.best_score:
            self.stats.best_score = self.score

        self.stats.epsilon_values.append(self.epsilon)
        self.stats.scores.append(self.score)

        if len(self.stats.scores) % 100 == 0 and len(self.stats.scores) != 0:
            print(f"Episode: {len(self.stats.scores)}, Total Best: {self.stats.best_score}, Last 100 Best: {max(self.stats.scores[-100:-1])}, Last 100 Averge: {np.average(self.stats.scores[-100:-1]):.1f}, Last 100 Min: {min(self.stats.scores[-100:-1]):.1f}, Epsilon: {self.epsilon:.3f}")

        self.epsilon_deacy()

        self.last_state = None
        self.last_action = None
        self.score = 0

            
    def plot_stats(self):

        fig, ax = plt.subplots()
        ax.set_xlabel('Episodes')
        ax2 = ax.twinx()

        # Score
        ax.plot(self.stats.scores, linestyle=' ', marker='.', color='#887aff', label="Scores")
        ax.set_ylabel('Score', color="blue")
        ax.tick_params(axis='y', colors="blue")

        # Moving Average Score
        ax.plot(moving_avg(self.stats.scores, len(self.stats.scores) / 10), color="blue", label="Moving Average")
        
        # Epsilon
        ax2.plot(self.stats.epsilon_values, color="orange")
        ax2.set_ylabel('Epsilon', color="orange")
        ax2.tick_params(axis='y', colors="orange")

        # Stop Learning
        for x in self.stats.stop_learning:
            plt.axvline(x, color = 'red', label = 'Stopped learning')
            plt.text(x, 5, "Stopped\nlearning", rotation=0, verticalalignment='center')

        # Legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

        plt.show()

    def policy(self, observation) -> int:

        state = self.get_state(observation)

        random = np.random.uniform()

        if self.keep_learning and random < self.epsilon:
            action = np.random.randint(2)
        else:
            action = np.argmax(self.q_table[state])
            
        self.last_state = state
        self.last_action = action

        return action

    def learn(self, observation, reward, score):
        self.score += score
        if not self.keep_learning: return

        current_q = self.q_table[self.last_state][self.last_action]
        max_future_q = np.max(self.q_table[self.get_state(observation)])
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_future_q)
        self.q_table[self.last_state][self.last_action] = new_q

