from abc import ABC, abstractmethod
import numpy as np

class Policy(ABC):
    """
    Abstract base class for a policy.
    """

    @abstractmethod
    def action_prob(self, state, action) -> float:
        """
        Returns the probability of taking the action if we are in the given state.

        Parameters:
            state: state
            action: action

        Returns:
            float: the probability of taking the action in the given state   
        """
        raise NotImplementedError()

    @abstractmethod
    def action(self, state) -> int | np.ndarray:
        """
        Chooses the action to take based on the policy.

        Parameters:
            state: state
        
        Returns:
            int | np.ndarray: the action(s) to take
        """
        raise NotImplementedError()

## This is a random policy that takes a static probability distribution over all actions.
## If none is specified, all actions can be taken randomly.
class RandomPolicy(Policy):
    def __init__(self, nA, p=None):
        self.p = p if p is not None else np.array([1/nA]*nA)

    def action_prob(self, state, action=None):
        return self.p[action]

    def action(self, state):
        return np.random.choice(len(self.p), p=self.p)
    
    def __str__(self) -> str:
        return f"RandomPolicy, p={self.p}"