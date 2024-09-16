from abc import ABC, abstractmethod
from typing import override
import numpy as np

from interfaces.solver import Hyperparameters, Solver
from assignments.bandit_policy import BanditPolicy
from assignments.bandit_env import BanditEnvironment

class ActionValue(ABC):
    """
    This is an abstract class. It represents the estimated action values (Q_hat) of a k-armed bandit.

    It provides a method to update the Q values based on the observed reward, and a method to select an action based on the estimated Q values.
    Do not instantiate this class directly, use one of the subclasses below instead.
    """

    def __init__(self, k: int):
        """
        Parameters:
            k (int): the number of bandit arms
        """
        self.k = k
        """The number of bandit arms."""
        self.Q_hat = np.zeros(k)
        """The estimated Q values for each action."""

    def reset(self) -> None:
        """
        Resets the estimated Q values to zero.

        Does not change the reference for Q_hat, just sets all values to zero.
        We don't change the references because the policy uses the same Q_hat reference.
        """
        self.Q_hat.fill(0.0)

    @abstractmethod
    def update(self, a: int, r: float) -> None:
        """
        Updates the estimated Q value for the selected action based on the observed reward.

        This method should be implemented by the subclasses.

        Parameters:
            a (int): the selected action
            r (float), the observed reward
        """
        raise NotImplementedError

class SampleAverageMethod(ActionValue):
    """
    Represents the estimated action values (Q) of a k-armed bandit using the sample-average method.

    The sample-average method updates the Q values by averaging every observed reward, for each action.
    """

    def __init__(self, k: int):
        """
        Parameters:
            k (int): the number of bandit arms
        """
        super().__init__(k)

        self.n = np.zeros(self.k)
        """The number of times each action has been selected."""

    @override
    def reset(self):
        """
        Resets the estimated Q values and the number of times each action has been selected to zero.
        """
        super().reset()
        self.n = np.zeros(self.k)

    def update(self, a: int, r: float):
        """
        Updates the estimated Q value for the selected action using the sample-average method.

        Parameters:
            a (int): the selected action
            r (float): the observed reward        
        """
        ### TODO ###
        ### 1. Update the number of times the selected action has been selected.
        ### 2. Update the estimated Q value for the selected action using the sample-average method (see equation 2.3)
        raise NotImplementedError


class ConstantStepSizeMethod(ActionValue):
    def __init__(self, alpha: float, k: int):
        super().__init__(k)
        self.alpha = alpha

    @override
    def update(self, a: int, r: float) -> None:
        """
        Updates the estimated Q value for the selected action using the constant step-size method.

        Parameters:
            a (int): the selected action
            r (float): the observed reward
        """
        ### TODO ###
        ### 1. Update the estimated Q value for the selected action using the constant step-size method (see equation 2.5)
        raise NotImplementedError

class BanditSolverHyperparameters(Hyperparameters):
    """Hyperparameters for the bandit solver."""
    def __init__(self, epsilon: float, alpha: float = None):
        self.gamma = None  # Unused for bandits

        self.epsilon = epsilon
        """The probability of selecting a random action."""
        self.alpha = alpha
        """If we are using ConstantStepSizeMethod, this is the step size parameter."""
        

class BanditSolver(Solver):
    def __init__(self, env: BanditEnvironment, hyperparameters: BanditSolverHyperparameters, Method: type[ActionValue]):
        super().__init__("Bandit", env, hyperparameters)

        if Method == SampleAverageMethod:
            self.method = Method(k = env.action_space.n)
        else:
            self.method = Method(k = env.action_space.n, alpha = hyperparameters.alpha)
            
        # We share the reference to the Q_hat function
        self.policy = BanditPolicy(
            Q = self.method.Q_hat,
            epsilon = self.hyperparameters.epsilon
        )

    @override
    def action(self, _: int) -> int:
        """
        Selects an action using the agent's policy.
        
        Parameters:
            _ (int): Unused, it would be the state, but bandits don't have states.
        """
        return self.policy.action(_)

    @override
    def train_episode(self):
        """
        Runs a single experiment with the given bandit and agent for the specified number of steps.
        Returns the rewards obtained at each step and whether the best action was taken at each step.

        Parameters:
            bandit (BanditEnvironment): The bandit to run the experiment on.
            agent (Policy_EpsilonGreedy): The agent to use for selecting actions and tracking Q values.
            steps (int): The number of steps to run the experiment for.
        
        Returns:
            tuple: A tuple containing:
                - rs (np.ndarray): The rewards obtained at each step.
                - best_action_taken (np.ndarray): Whether the best action was taken at each step.
        """
        env: BanditEnvironment = self.env
        """The bandit environment to run the experiment on."""
        policy: BanditPolicy = self.policy
        """The policy to use for selecting actions."""
        method: ActionValue = self.method
        """The method to use for estimating the Q values (e.g., SampleAverageMethod or ConstantStepSizeMethod)."""

        # Reset everything to a good starting state
        _, _ = self.env.reset()
        done = truncated = False
        method.reset()

        rs = []
        best_action_taken = []

        while not done and not truncated:
            ### TODO ###
            ### 1. Select an action using the agent's policy
            ### 2. Step the bandit environment with the selected action
            ###     Hint: make sure to destructure the step method's return values (next_state, r, done, truncated, info)
            ### 3. Update the Q value using the appropriate method

            rs.append(r)
            best_action_taken.append(info["ideal_action"])

        return np.array(rs), np.array(best_action_taken)
