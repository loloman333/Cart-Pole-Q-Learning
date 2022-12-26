from q_learner import q_learner

ENV_NAME = "CartPole-v1"

class cartpole_learner(q_learner):

    def __init__(self, epsilon: float, alpha: float, gamma: float, epsilon_change: float, epsilon_min: float) -> None:
        super().__init__(441, 2, epsilon, alpha, gamma, epsilon_change, epsilon_min)
    
    def get_state_simple(self, observation):

        pole_angle = observation[2]
        pole_angle_state_count = 21          
        pole_angle_state = self.get_substate(pole_angle, pole_angle_state_count, 0.418 * 2 / pole_angle_state_count)

        pole_velocity = observation[3]    
        pole_velocity_state_count = 21      
        pole_velocity_state = self.get_substate(pole_velocity, pole_velocity_state_count, 4 * 2 / pole_velocity_state_count)

        return self.combine_substates(
            [pole_angle_state, pole_velocity_state],
            [pole_angle_state_count, pole_velocity_state_count]
        )

    def get_state(self, observation):
        state = self.get_state_simple(observation)
        assert (state >= 0 or state < self.num_states), f"{state}"
        return state

