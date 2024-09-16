import numpy as np
from typing import override

from interfaces.policy import Policy

class BanditPolicy(Policy):
    """
    Implements a simple epsilon-greedy policy.

    (We only use this policy for the Bandits environment.)
    """
    def __init__(self, Q: np.ndarray, epsilon: float):
        """
        Parameters:
            Q (np.ndarray): The estimated action (Q-hat) values. Shape (nA).
            epsilon (float): The probability of selecting a random action.
        """
        self.epsilon = epsilon
        self.Q = Q

    @override
    def action(self, _: int = None) -> int:
        """
        Selects an action based on an epsilon-greedy policy.

        Parameters:
            _ (int): Unused. Just here to conform to interface, will always be None for bandits.
        """
        ### TODO ###
        ### 1. Implement the epsilon-greedy policy for selecting an action.
        ###    With probability epsilon, select a random action.
        ###    Otherwise, select any action with the highest estimated Q value.
        raise NotImplementedError

    @override
    def action_prob(self, _: int = None, action: int = None) -> float:
        """
        Returns the probability of taking the action if we are in the given state.

        This is not used for bandits but we provide the solution for completeness.
        
        Parameters:
            _ (int): unused. would be the state index, but bandits don't have states.
            action (int): action index

        Returns:
            float: the probability of taking the action
        """
        assert action is not None, "action must be provided"
        assert _ is None or _ == 0, "Bandits don't have states, so the state index must be None or 0"
        return self.epsilon * (1 / len(self.Q)) + (1 - self.epsilon) * float(self.Q.argmax() == action)

