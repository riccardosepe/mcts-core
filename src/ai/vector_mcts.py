import random

import numpy as np

from ..tree.tree import Tree, Node
from ..ai.mcts import MCTS

class VMCTS(MCTS):
    def _build_tree(self):
        root_data = self.transition_model.backup()
        if root_data['reward'] is None:
            root_data['reward'] = np.zeros(self.transition_model.reward_space_cardinality)
        tree = Tree(self.transition_model.legal_actions, root_data, adversarial=self.adversarial)
        assert isinstance(tree.root.game_reward, np.ndarray), type(tree.root.game_reward)
        tree.root.features = tree.root.game_reward
        return tree

    def _rand_rollout_eval(self, leaf_node):
        if leaf_node.is_terminal:
            return [np.zeros_like(leaf_node.game_reward)]
        ret = []
        d = False
        while not d:
            action = random.choice(self.transition_model.legal_actions)
            _, r, d, _, _ = self.transition_model.step(action)
            ret.append(r * self.gamma)
            self.t += 1
            if self.transition_model.t >= self._max_depth:
                break
        return [sum(ret)]

    def _evaluate(self, leaf_node):
        collected_rewards = self._rand_rollout_eval(leaf_node)
        assert isinstance(leaf_node.game_reward, np.ndarray)
        leaf_node.features = leaf_node.game_reward
        return collected_rewards

    def _backpropagate(self, node, score, terminal_node):
        if node is None:
            return
        node: Node
        score = [self.gamma * s for s in score]
        score.append(node.game_reward)
        node.update_score(sum(sum(score))) # sum over time and over reward components
        node.visit()
        node.update_subtree(terminal_node)
        if self._evaluator is not None:
            #  Update the features of the subtree of the node based on the newly added terminal node's features
            node.update_subtree_features(terminal_node.features)
        # the features coming from the rollout are needed only by the root to motivate its action selection
        # NB: optionally for the future it might be needed inside the node evaluated with the rollout as well
        else:
            if node.is_root:
                # do nothing for the root subtree features, they are only needed for the 1st generation of the root
                pass
            elif node.parent.is_root:
                node.update_subtree_features(sum(score)) # sum over time but not over reward components
            else:
                node.update_subtree_features(terminal_node.features)

        self._backpropagate(node.parent, score, terminal_node)

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
        if not selected_node.is_terminal:
            # 2. EXPAND
            expanded_node = self._expand(selected_node, selected_action)
            terminal_node = expanded_node

            # 3. EVALUATE
            score = self._evaluate(expanded_node)

        else:
            terminal_node = selected_node
            score = [np.zeros_like(terminal_node.game_reward)]

        # 4. BACKPROPAGATE
        self._backpropagate(terminal_node, score, terminal_node)

        # restore the game state
        self.transition_model.load(checkpoint)

    def _check_explainer(self):
        return