from matplotlib import pyplot as plt
from lib.run_bandit import run_bandit

def visualize(num_simulations: int, num_iterations: int, k: int):
    results = run_bandit(
        num_simulations = num_simulations,
        num_iterations = num_iterations,
        k = k
    )
    sample_average = {
        'average_rs': results[0],
        'average_best_action_taken': results[1],
    }
    constant = {
        'average_rs': results[2],
        'average_best_action_taken': results[3],
    }

    assert len(sample_average['average_rs']) == len(sample_average['average_best_action_taken']) == \
        len(constant['average_rs']) == len(constant['average_best_action_taken']) == num_iterations

    fig,axes = plt.subplots(2,1)

    axes[1].set_ylim([0.,1.])

    axes[0].plot(sample_average['average_rs'], label="sample average")
    axes[0].plot(constant['average_rs'], label="constant step-size")
    axes[0].set_ylabel("reward")
    axes[0].legend()

    axes[1].plot(sample_average['average_best_action_taken'], label="sample average")
    axes[1].plot(constant['average_best_action_taken'], label="constant step-size")
    axes[1].set_xlabel("# time steps")
    axes[1].set_ylabel("best action taken")
    axes[1].legend()

    plt.show(block=True)