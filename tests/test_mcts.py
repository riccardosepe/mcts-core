from unittest import TestCase
import unittest

from ..src.ai import MCTS
from ..envs.tic_tac_toe import TicTacToeEnv


class TestMCTS(TestCase):

    def _test_counter_opponent_1(self, seed):
        env = TicTacToeEnv()

        env.reset(human_first=True, seed=seed)

        # build the following grid
        # O| |O
        #  |X|
        #  | |

        # human
        env.step(0)
        # bot
        env.step(4)
        # human
        env.step(2)

        agent = MCTS(env, seed=seed)

        action = agent.plan(iterations_budget=1000)

        try:
            assert action == 1
        except AssertionError as e:
            raise e

    def test_counter_opponent_1(self):
        c = []
        for i in range(10):
            try:
                self._test_counter_opponent_1(seed=i)
            except AssertionError:
                c.append(i)
        try:
            assert len(c) == 0
        except AssertionError as e:
            print(c)
            raise e

    def _test_win_1(self):
        env = TicTacToeEnv()
        env.reset(human_first=True, seed=0)

        # build the following grid
        # O|O|X
        # O| |X
        #  | |

        # human
        env.step(0)
        # bot
        env.step(2)
        # human
        env.step(1)
        # bot
        env.step(5)
        # human
        env.step(3)

        agent = MCTS(env)
        action = agent.plan(iterations_budget=1000)

        assert action == 8

    def test_win_1(self):
        c = []
        for i in range(10):
            try:
                self._test_win_1()
            except AssertionError:
                c.append(i)

        assert len(c) == 0



if __name__ == '__main__':
    unittest.main()
