import gymnasium as gym
import numpy as np

class BanditEnvironment(gym.Env):
    """
    Represents a k-armed bandit problem as a gym environment.
    """

    def __init__(self, k: int):
        """
        Parameters:
            k (int): the number of bandit arms
        """
        self.action_space = gym.spaces.Discrete(k)
        self.observation_space = None # for bandits we don't use states
        self.Q_star = np.zeros((self.action_space.n))

    def step(self, a: int) -> tuple[float, bool]:
        """
        Performs a step of the bandit.

        Parameters:
            a (int): the selected action to draw a sample from.

        Returns:
            tuple: A tuple containing:
                - next_state (None): Always None for bandit problems, since there is no state.
                - reward (float): The reward of the selected bandit arm.
                - done (False): Always False for bandit problems, since there is no state.
                - truncated (False): False for bandits, although the TimeLimit wrapper will set this to True.
                - info (dict): A dictionary with the following keys:
                    - "ideal_action" (bool): True if the selected arm was one of the best
        """
        assert a < len(self.Q_star), f"Invalid action: {a}"
        reward = self._sample_reward(a)
        ideal_action = self._is_ideal_action(a)
        self._walk_all_arms()
        return None, reward, False, False, {"ideal_action": ideal_action}

    def reset(self) -> None:
        """
        Resets the bandit to its initial state.
        """
        self.Q_star = np.zeros_like(self.Q_star)

        return None, {}

    def _sample_reward(self, action: int) -> float:
        """
        Samples a reward from the selected arm. r ~ N(Q*(a), 1.0)

        Parameters:
            action (int): the selected action
        """
        ### TODO ###
        ### 1. Sample a reward from the selected arm.
        #       Hint: use np.random.normal()
        #       Hint: np.random.normal() accepts the mean and standard deviation, not the variance.
        raise NotImplementedError

    def _is_ideal_action(self, action: int) -> bool:
        """
        Determines if the selected action is one of the best actions.
        
        Parameters:
            action (int): the selected action
            
        Returns:
            bool: True if the selected action is one of the best actions.
        """
        ### TODO ###
        ### 1. Determine if the selected action is one of the best actions.
        ###    Hint: remember that multiple arms could have the same Q* value (e.g. on the first step, all arms are valid)
        raise NotImplementedError


    def _walk_all_arms(self):
        """
        Walks all the arms of the bandit, adding noise to the true Q* values.

        noise ~ N(0, 0.01)
        """
        ### TODO ###
        ### 1. Add noise to all the arms of the bandit.
        #       Hint: use np.random.normal() again, but with the size parameter this time.
        raise NotImplementedError
