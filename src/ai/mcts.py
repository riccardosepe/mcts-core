import os
import pickle
import random
import sys
import time
from datetime import datetime
from functools import cmp_to_key

import numpy as np
from tqdm import tqdm

from ..tree import Tree, Node
from ..tree.chance_tree import ChanceNode


class MCTS:
    # Add 2 visits and 1 victory to get average value
    DUMMY_NODE = Node.construct_dummy_node(score=1, visits=2)

    def __init__(self,
                 transition_model,
                 adversarial=True,
                 gamma=1,
                 keep_subtree=True,
                 max_depth=1000,
                 use_tqdm=False,
                 exploration_constant=None,
                 schedule_exploration=False,
                 evaluator_cls=None,
                 explainer_cls=None,
                 seed=None,
                 pw=False,
                 **kwargs):

        self.transition_model = transition_model
        self._max_depth = max_depth if max_depth else np.inf
        self._build_evaluator(evaluator_cls, **kwargs)
        self._build_explainer(explainer_cls, **kwargs)
        self.adversarial = adversarial
        self.tree = None
        self.gamma = gamma
        self._keep_subtree = keep_subtree
        self.seed = seed

        self.t = None
        if exploration_constant is None:
            self._exploration_constant = np.sqrt(2)
        else:
            self._exploration_constant = exploration_constant

        self._schedule_exploration = schedule_exploration

        self._use_tqdm = use_tqdm

        self._check_explainer()

        self._pw = pw

        if self._pw:
            self._pw_k = kwargs.pop('pw_k', 1.)
            self._pw_alpha = kwargs.pop('pw_alpha', .5)

        # random.seed(seed)
        # np.random.seed(seed)

    @property
    def exploration_constant(self):
        if self._schedule_exploration:
            return self._exploration_constant_schedule
        else:
            return self._exploration_constant

    @property
    def _exploration_constant_schedule(self):
        return self._exploration_constant * (self._max_depth - self.t + 1) / self._max_depth

    def _build_evaluator(self, evaluator_cls, **kwargs):
        if evaluator_cls is None:
            self._evaluator = None
        else:
            self._evaluator = evaluator_cls(self.transition_model, **kwargs)

    def _check_explainer(self):
        assert not (self._evaluator is None and self._explainer is not None)

    def _build_explainer(self, explainer_cls, **kwargs):
        if explainer_cls is None:
            self._explainer = None
        else:
            self._explainer = explainer_cls(transition_model=self.transition_model, **kwargs)

    def reset(self):
        self.t = None
        self.tree = None

    def step_consistency(self, action=None):
        if self._keep_subtree:
            return
        else:
            self.reset()

    def _build_tree(self):
        tree = Tree(self.transition_model.legal_actions, self.transition_model.backup(), adversarial=self.adversarial)
        if self._evaluator is not None:
            self._evaluate(tree.root)
        return tree

    def _select(self):
        node = self.tree.root
        while not node.is_terminal:
            child = self.select_ucb(node)
            if child is self.DUMMY_NODE:
                return node, child.action
            elif child.time >= self._max_depth:
                return node, child.action
            self.transition_model.step(child.action)
            self.t += 1
            node = child
        return node, None

    def _expand(self, node, action=None):
        if action is None:
            action = node.random_action()

        if self._pw:
            max_children = int(self._pw_k * (node.visits ** self._pw_alpha))
            if len(node.branched_children) >= max_children:
                action = random.choice(node.branched_children)
        try:
            self.transition_model.step(action)
        except Exception as e:
            print(e, file=sys.stderr)
            env_bkp = self.transition_model.backup()
            bot_bkp = self.backup()
            backup = {
                'env': env_bkp,
                'bot': bot_bkp,
                'seed': self.seed,
                'exploration_constant': self.exploration_constant
            }

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("dumps", exist_ok=True)
            with open(f'dumps/dump_{self.seed}_{timestamp}.pkl', 'wb') as dumpfile:
                pickle.dump(backup, dumpfile)
            raise e

        if node.need_to_expand(action):
            new_node = self.tree.insert_node(node.id,
                                             action,
                                             self.transition_model.legal_actions,
                                             self.transition_model.backup())
        else:
            new_node = node.children[action]
            assert new_node is not None
        self.t += 1

        return new_node

    def _rand_rollout_eval(self, leaf_node):
        if leaf_node.is_terminal:
            return leaf_node.game_reward
        ret = 0
        d = False
        while not d:
            action = random.choice(self.transition_model.legal_actions)
            _, r, d, _, _ = self.transition_model.step(action)
            # sparse / non-sparse setting
            # in sparse setting, non-terminal rewards are 0
            ret = + r * self.gamma
            self.t += 1
            if self.transition_model.t >= self._max_depth:
                # TODO: handle this better
                # ret = 0
                break
        return ret

    def _evaluate(self, leaf_node):
        if self._evaluator is None:
            v_hat = self._rand_rollout_eval(leaf_node)
            if self.adversarial:
                assert -1 <= v_hat <= 1
                v_hat = (v_hat + 1) / 2
            value = v_hat
        else:
            node_state = leaf_node.game_state
            node_time = leaf_node.time
            terminal_reward = leaf_node.game_reward
            features = self._evaluator.evaluate(node_state, node_time, terminal_reward=terminal_reward)

            # features is a 1D array: [terminal_reward_component, h_1, ..., h_M]
            features = np.concatenate((np.atleast_1d(terminal_reward), features))
            # value == features.sum() by construction (reward slot + heuristic slots)
            D = features.size
            value = (features.sum() + 1) / 2
            # shift and rescale each entry so that features.sum() == value
            features = (features + 1 / D) / 2

            if self.adversarial and leaf_node.player == 'Agent':
                value = 1 - value
                features = 1 / D - features

            leaf_node.features = features

        return value

    def _backpropagate_iter(self, node, score):
        while node is not None:
            node.visit()
            node.update_score(score)
            node = node.parent

    def _backpropagate(self, node, score, terminal_node, terminal_node_features):
        if node is None:
            return
        node: Node
        if not node.is_terminal:
            score += node.game_reward
        node.update_score(score)
        node.visit()
        node.update_subtree(terminal_node)
        node.update_distribution()
        if self._evaluator is not None:
            #  Update the features of the subtree of the node based on the newly added terminal node's features
            node.update_subtree_features(terminal_node_features)
        if self.adversarial:
            new_score = 1 - score
            if self._evaluator is not None:
                D = terminal_node_features.size
                terminal_node_features =  1 / D - terminal_node_features
        else:
            # sparse / non-sparse setting
            # in sparse setting, non-terminal rewards are 0
            new_score = score * self.gamma
        self._backpropagate(node.parent, new_score, terminal_node, terminal_node_features)

    def _plan_iteration(self):
        """
        The core of the MCTS algorithm, i.e. the sequence of the four steps: Select, Expand, Simulate, Backpropagate.
        """
        # save the game state
        checkpoint = self.transition_model.backup()

        self.t = 0

        # 1. SELECT
        selected_node, selected_action = self._select()

        # NB: very uncommon in practice, the following lines handle small game trees where it's possible to reach a
        #  terminal state during the expansion phase

        # TODO: IS THIS THE BETTER WAY TO ACHIEVE THIS?
        if isinstance(selected_node, ChanceNode) or not selected_node.is_terminal:
            # 2. EXPAND
            expanded_node = self._expand(selected_node, selected_action)
            terminal_node = expanded_node

            # 3. EVALUATE
            value = self._evaluate(expanded_node)

        else:
            terminal_node = selected_node
            value = terminal_node.value

        # 4. BACKPROPAGATE
        terminal_node_features = terminal_node.features
        self._backpropagate(terminal_node, value, terminal_node, terminal_node_features)

        # restore the game state
        self.transition_model.load(checkpoint)

    def _plan(self, iterations_budget=None, time_budget=None):
        """
        Run a bunch of `_plan_iteration`s until either the iterations budget or the time budget is reached.

        :param iterations_budget: the maximum number of iterations to run
        :param time_budget: the maximum available time for a single action (in seconds)
        :return: the chosen action
        """

        if self._use_tqdm and iterations_budget is not None:
            progress_bar = tqdm(total=iterations_budget)
        else:
            progress_bar = None

        if iterations_budget is None and time_budget is None:
            raise ValueError("Either iterations_budget or time_budget must be set")
        elif iterations_budget is None:
            iterations_budget = np.inf
        elif time_budget is None:
            time_budget = np.inf

        elapsed_time = 0
        iteration = 0

        start_time = time.perf_counter()

        while elapsed_time < time_budget and iteration < iterations_budget:
            self._plan_iteration()
            elapsed_time = (time.perf_counter() - start_time) * 1000
            iteration += 1
            if progress_bar:
                progress_bar.update(1)
        best_child = self.root_best_child()
        return best_child

    def plan(self, iterations_budget=None, time_budget=None, explain=None):
        """
        Wrapper for the actual _plan method.
        """
        if explain is None:
            explain = False
            confidence = None
        else:
            if isinstance(explain, bool):
                explain = explain
                confidence = None
            else:
                try:
                    confidence = float(explain)
                except ValueError:
                    raise ValueError("Invalid value for explain parameter. Must be a boolean or a float.")
                explain = True

        if self.tree is None:
            self.tree = self._build_tree()
        if explain:
            assert self._explainer is not None
        explanation = None
        if isinstance(self.tree.root, ChanceNode):
            raise RuntimeError
        best_child = self._plan(iterations_budget=iterations_budget, time_budget=time_budget)
        best_action = best_child.action
        if explain:
            explanation = self._explainer.explain(self.tree, best_action, confidence=confidence)

        if self._keep_subtree:
            self.tree.keep_subtree(best_child)
        if explain:
            return best_action, explanation
        else:
            return best_action, None

    def init_tree(self, legal_actions, root_data):
        # TODO: CHECK THIS
        self.tree = Tree(root_legal_actions=legal_actions, root_data=root_data, adversarial=self.adversarial)

    def opponent_action(self, action):
        if not self.adversarial:
            return
        if self.tree.root.is_leaf:
            # self.tree.root.ply(action)
            self.reset()
        else:
            new_root = self.tree.root.children[action]
            self.tree.keep_subtree(new_root)

    @staticmethod
    def uct(node, parent, c=np.sqrt(2)):
        """
        Calculates the Upper Confidence Bound for a tree.
        :param node: the node for which it calculates the UCB
        :param parent: the parent node of `node`
        :param c: the coefficient of the formula
        :return: (exploitation, exploration) tuple, both in [-1, 1] space
        """
        exploitation = node.score / node.visits
        if parent.visits == 0:
            exploration = 0
        else:
            exploration = np.sqrt(
                np.log(parent.visits) / node.visits
            )
        return exploitation + c * exploration

    def select_ucb(self, parent):
        children = list(parent.children.items())
        random.shuffle(children)
        scores = [(idx, MCTS.uct(node if node else self.DUMMY_NODE, parent, self.exploration_constant)) for idx, node in
                  children]

        max_score = max(s for _, s in scores)
        epsilon = abs(max_score) * 1e-10
        best_candidates = [idx for idx, s in scores if abs(s - max_score) <= epsilon]
        best_action = random.choice(best_candidates)
        if parent.children[best_action] is not None:
            child = parent.children[best_action]
        else:
            child = self.DUMMY_NODE
            child.action = best_action
        return child

    def root_best_child(self):
        children_list = list(filter(lambda n: n is not None, self.tree.root.children.values()))
        assert len(children_list) > 0
        return sorted(children_list, key=cmp_to_key(Node.node_cmp))[0]

    def backup(self):
        bkp = {
            'tree': self.tree.backup(),
            't': self.t,
        }
        return bkp

    def load(self, bkp):
        self.tree.load(bkp['tree'])
        self.t = bkp['t']
