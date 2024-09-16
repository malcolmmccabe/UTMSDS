from tqdm import tqdm
import numpy as np
from gymnasium.wrappers import TimeLimit

from assignments.bandit import BanditEnvironment, SampleAverageMethod, ConstantStepSizeMethod, BanditSolver, BanditSolverHyperparameters

def run_bandit(num_simulations: int = 300, num_iterations: int = 10000, k: int = 10):
    """
    Runs the bandit experiment with the sample average and constant step-size methods.

    Parameters:
        num_simulations (int): The number of bandit runs to average over.
        num_iterations (int): The number of steps to run each bandit for.
        k (int): The number of bandits.

    Returns:
        tuple: A tuple containing:
            - average_rs_sample_average (np.ndarray): The average reward over all bandit runs for the sample average method.
            - average_best_action_taken_sample_average (np.ndarray): The average best action taken over all bandit runs for the sample average method.
            - average_rs_constant (np.ndarray): The average reward over all bandit runs for the constant step-size method.
            - average_best_action_taken_constant (np.ndarray): The average best action taken over all bandit runs for the constant step-size method.
    """
    sample_average_solver = BanditSolver(
        env = TimeLimit(BanditEnvironment(k=k), max_episode_steps=num_iterations),
        Method = SampleAverageMethod,
        hyperparameters = BanditSolverHyperparameters(
            epsilon = 0.1
        )
    )

    constant_step_solver = BanditSolver(
        env = TimeLimit(BanditEnvironment(k=k), max_episode_steps=num_iterations),
        Method = ConstantStepSizeMethod,
        hyperparameters = BanditSolverHyperparameters(
            epsilon = 0.1,
            alpha = 0.1
        )
    )
    
    outputs = []

    for agent in [sample_average_solver, constant_step_solver]:
        average_rs, average_best_action_taken = [], []

        for _ in tqdm(range(num_simulations)):
            Rs, best_action_taken = agent.train_episode()
            
            agent.method.reset()
            
            average_rs.append(Rs)
            average_best_action_taken.append(best_action_taken)

        average_rs = np.mean(np.array(average_rs), axis=0)
        average_best_action_taken = np.mean(np.array(average_best_action_taken), axis=0)

        outputs += [average_rs, average_best_action_taken]

    return outputs[0], outputs[1], outputs[2], outputs[3]
