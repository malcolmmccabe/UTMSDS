import unittest
import numpy as np
from lib.run_bandit import run_bandit
from assignments.bandit_env import BanditEnvironment
from assignments.bandit_policy import BanditPolicy
from assignments.bandit import SampleAverageMethod, ConstantStepSizeMethod

class TestBanditEnvironment(unittest.TestCase):
    """BanditEnvironment"""

    def test_constructor(self):
        """BanditEnvironment()"""
        self.assertTrue(np.array_equal(BanditEnvironment(5).Q_star.shape, (5,)), f"Q_star should be a 1D array. Expected shape: (5), but got: {BanditEnvironment(5).Q_star.shape}")
    
    def test_reset(self):
        """BanditEnvironment.reset()"""
        bandit = BanditEnvironment(k=1)
        bandit.Q_star = np.ones_like(bandit.Q_star)
        bandit.reset()
        self.assertTrue(np.array_equal(bandit.Q_star, np.zeros_like(bandit.Q_star)))

    def test_sample_reward(self):
        """_sample_reward() - reward ~ N(q*, 1)"""
        bandit = BanditEnvironment(k=1)
        np.random.seed(0)
        bandit.Q_star = np.array([100.0])
        r = bandit._sample_reward(0)
        self.assertEqual(r, 101.76405234596767)
        
    def test_step_is_ideal_action(self):
        """_is_ideal_action()"""
        bandit = BanditEnvironment(k=3)

        bandit.Q_star = np.array([0.0, 100.0, 100.0])
        ideal_action = bandit._is_ideal_action(0)
        self.assertFalse(ideal_action, "The first action is not the ideal action")

        bandit.Q_star = np.array([0.0, 100.0, 100.0])
        ideal_action = bandit._is_ideal_action(1)
        self.assertTrue(ideal_action, "Either of the two actions can be the ideal action")

        bandit.Q_star = np.array([0.0, 100.0, 100.0])
        ideal_action = bandit._is_ideal_action(2)
        self.assertTrue(ideal_action, "Either of the two actions can be the ideal action")
        
    def test_walk_all_arms(self):
        """BanditEnvironment.walk_all_arms() - noise ~ N(0, 0.01)"""
        bandit = BanditEnvironment(k=2)
        np.random.seed(0)

        bandit.Q_star = np.array([100.0, 0.0])
        bandit._walk_all_arms()

        expected = np.array([1.00017641e+02, 4.00157208e-03])
        self.assertTrue(
            np.allclose(bandit.Q_star, expected, 1e-8),
            f"expected: {expected} but got: {bandit.Q_star}"
        )

        
class TestSampleAverageMethod(unittest.TestCase):
    """SampleAverageMethod"""

    def test_reset(self):
        """SampleAverageMethod.reset()"""
        av = SampleAverageMethod(5)
        av.Q_hat.fill(1.0)
        av.n = np.ones(5)
        av.reset()
        self.assertTrue(np.array_equal(av.Q_hat, np.zeros((5))))
        self.assertTrue(np.array_equal(av.n, np.zeros(5)))

    def test_update(self):
        """SampleAverageMethod.update()"""
        sa = SampleAverageMethod(k = 5)
        sa.Q_hat = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        sa.n = np.array([0, 0, 0, 0, 0])

        sa.update(0, 1.0)
        self.assertTrue(np.allclose(sa.Q_hat, np.array([1.0, 0.0, 0.0, 0.0, 0.0]), 1e-8))
        self.assertTrue(np.allclose(sa.n, np.array([1, 0, 0, 0, 0]), 1e-8))

        sa.update(0, 2.0)
        self.assertTrue(np.allclose(sa.Q_hat, np.array([1.5, 0.0, 0.0, 0.0, 0.0]), 1e-8))
        self.assertTrue(np.allclose(sa.n, np.array([2, 0, 0, 0, 0]), 1e-8))

        sa.update(1, 1.0)
        self.assertTrue(np.allclose(sa.Q_hat, np.array([1.5, 1.0, 0.0, 0.0, 0.0]), 1e-8))
        self.assertTrue(np.allclose(sa.n, np.array([2, 1, 0, 0, 0]), 1e-8))

        sa.update(1, 2.0)
        self.assertTrue(np.allclose(sa.Q_hat, np.array([1.5, 1.5, 0.0, 0.0, 0.0]), 1e-8))
        self.assertTrue(np.allclose(sa.n, np.array([2, 2, 0, 0, 0]), 1e-8))

class TestConstantStepSizeMethod(unittest.TestCase):
    """ConstantStepSizeMethod"""

    def test_reset(self):
        """ActionValue.reset()"""
        av = SampleAverageMethod(5)
        av.Q_hat = np.ones(5)
        av.reset()
        self.assertTrue(np.array_equal(av.Q_hat, np.zeros(5)))

    def test_update(self):
        """ConstantStepSizeMethod.update()"""
        css = ConstantStepSizeMethod(alpha = 0.1, k = 5)
        css.Q_hat = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        css.update(0, 1.0)
        expected = np.array([0.1, 0.0, 0.0, 0.0, 0.0])
        self.assertTrue(np.allclose(css.Q_hat, expected, 1e-8), f"Expected: {expected} but got: {css.Q_hat}")

        css.update(0, 2.0)
        expected = np.array([0.29, 0.0, 0.0, 0.0, 0.0])
        self.assertTrue(np.allclose(css.Q_hat, expected, 1e-8), f"Expected: {expected} but got: {css.Q_hat}")

        css.update(1, 1.0)
        expected = np.array([0.29, 0.1, 0.0, 0.0, 0.0])
        self.assertTrue(np.allclose(css.Q_hat, expected, 1e-8), f"Expected: {expected} but got: {css.Q_hat}")

        css.update(1, 2.0)
        expected = np.array([0.29, 0.29, 0.0, 0.0, 0.0])
        self.assertTrue(np.allclose(css.Q_hat, expected, 1e-8), f"Expected: {expected} but got: {css.Q_hat}")

class TestBanditPolicy(unittest.TestCase):
    """BanditPolicy"""
    k = 5
    def test_action(self):
        """BanditPolicy.action()"""
        random_agent = BanditPolicy(np.array([0.0, 0.0, 100.0, 0.0, 0.0]), 1.0)
        greedy_agent = BanditPolicy(np.array([0.0, 0.0, 100.0, 0.0, 0.0]), 0.0)

        # Random policy should return random actions
        np.random.seed(0)
        random_actions = [random_agent.action() for _ in range(25)]
        expected = np.array([0, 3, 3, 4, 0, 2, 0, 1, 1, 0, 0, 0, 3, 0, 1, 2, 3, 2, 2, 4, 0, 4, 2, 1, 1])
        self.assertTrue(np.equal(random_actions, expected).all())

        # Greedy policy should return the best action
        greedy_actions = [greedy_agent.action() for _ in range(100)]
        self.assertTrue(np.all(np.array(greedy_actions) == 2))

    def test_action_prob(self):
        """EpisilonGreedyBanditAgent.action_prob()"""
        agent = BanditPolicy(Q=np.array([0.0, 0.0, 100.0, 0.0, 0.0]), epsilon=0.1)
        self.assertAlmostEqual(agent.action_prob(action=0), 0.02)
        self.assertAlmostEqual(agent.action_prob(action=1), 0.02)
        self.assertAlmostEqual(agent.action_prob(action=2), 0.92)
        self.assertAlmostEqual(agent.action_prob(action=3), 0.02)
        self.assertAlmostEqual(agent.action_prob(action=4), 0.02)

class TestBanditSolver(unittest.TestCase):
    """BanditSolver"""
    def test_first_step_best_action(self):
        """The first step should always return 100% best action selection"""
        _, sample_avg_best_action, _, constant_best_action = \
            run_bandit(num_simulations=1000, num_iterations=1, k=10)
        
        self.assertTrue(np.all(sample_avg_best_action == 1.0), "Sample Average Method should always select the best action")
        self.assertTrue(np.all(constant_best_action == 1.0), "Constant Step Size Method should always select the best action")

    def test_first_step_rs(self):
        """The first step reward should be 0 in expectation, since we haven't done a walk yet"""
        sample_avg_rs, _, constant_rs, _ = \
            run_bandit(num_simulations=10000, num_iterations=1, k=10)
        
        self.assertAlmostEqual(np.mean(sample_avg_rs), 0.0, 1, "Sample Average Method should center around 0.0")
        self.assertAlmostEqual(np.mean(constant_rs), 0.0, 1, "Constant Step Size Method should center around 0.0")

if __name__ == '__main__':
    unittest.main()


    