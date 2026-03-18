"""
This file contains all the code relative to a tree data structure
"""
import random
from functools import cmp_to_key

import numpy as np

from .tree_visualizer import InteractiveTreeVisualizer


class Node:
    def __init__(self, parent_node, _id, legal_actions, game_data, action):
        self._id = _id
        self._children = dict.fromkeys(legal_actions, None)
        self._parent_node = parent_node
        self._available_actions = legal_actions[:]
        self._visits = 0
        self._score = 0
        self._game_data = game_data
        self._action = action
        self._distribution = dict.fromkeys(legal_actions, 0)
        self._probability = 0
        self._occupancy_frequency = 0
        self._features = None
        self._subtree = set()
        self._subtree_features = []
        self._is_confident = True

        # For confidence intervals
        self._score_squared = 0

    def __repr__(self):
        return f"{self.player}(id={self._id}, visits={self._visits}, value={self.value}, action={self._action})"

    def add_child(self, child):
        # NB: this method is only meant to be used within the Tree class
        self._children[child.action] = child
        self._available_actions.remove(child.action)

    def update_distribution(self):
        if self.is_leaf:
            return

        sum_visits = sum(map(lambda x: x.visits, filter(lambda n: n is not None, self._children.values())))
        self._distribution = {ch: n.visits / sum_visits for ch, n in self._children.items() if n is not None}
        for ch, n in self._children.items():
            if n is not None:
                n.probability = self._distribution[ch]

    def visit(self, n=1):
        self._visits += n

    def update_score(self, score):
        self._score += score
        self._score_squared += score**2

    def update_subtree(self, node):
        self._subtree.add(node)

    def update_subtree_features(self, features):
        if isinstance(features, dict):
            if 'value' in features:
                features.pop('value')
            features = list(features.values())
            feature_tuple = tuple(v.tolist() for v in features)
            self._subtree_features.append(feature_tuple)
        else:
            assert np.array(features).ndim == 1
            self._subtree_features.append(features)

    @property
    def q_vector(self):
        features = self.subtree_features
        if features is None:
            return None
        components_means = []
        for f in features:
            components_means.append(f.mean(axis=0))
        q1 = np.concatenate([*components_means])
        return q1

    def random_action(self):
        action = random.choice(self._available_actions)
        return action

    def set_root(self):
        assert self._parent_node is not None
        self._parent_node = None
        self._occupancy_frequency = 1
        self._probability = 1

    def ply(self, action):
        assert self.is_root
        self._available_actions.remove(action)
        del self._children[action]

    @property
    def best_child(self):
        children_list = list(filter(lambda n: n is not None, self._children.values()))
        return sorted(children_list, key=cmp_to_key(Node.node_cmp))[0]

    @property
    def probability(self):
        return self._probability

    @probability.setter
    def probability(self, value):
        self._probability = value

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, features):
        self._features = features

    @property
    def subtree(self):
        return self._subtree

    @property
    def subtree_features(self):
        if len(self._subtree_features) == 0:
            return None
        return list(map(np.array, zip(*self._subtree_features)))

    @property
    def occupancy_frequency(self):
        return self._occupancy_frequency

    @occupancy_frequency.setter
    def occupancy_frequency(self, value):
        self._occupancy_frequency = value

    @staticmethod
    def node_cmp(this, other):
        """
        Compares two nodes. The node with the highest number of visits is considered the best one.
        In case of ties, the node with the highest score is considered the best one.
        In case of further ties, the node with the highest sum of mean subtree features is considered the best one.
        In case of further ties, the nodes are considered equal and one is chosen randomly.
        1 means that `this` is worse than `other`
        -1 means that `this` is better than `other`
        0 means that they are equal
        """
        if this.value > other.value:
            return -1
        elif this.value < other.value:
            return 1
        else:
            if this.visits > other.visits:
                return -1
            elif this.visits < other.visits:
                return 1
            else:
                if this.subtree_features is not None and other.subtree_features is not None:
                    q1 = np.sum(np.mean(np.hstack(this.subtree_features), axis=0))
                    q2 = np.sum(np.mean(np.hstack(other.subtree_features), axis=0))
                    if q1 > q2:
                        return -1
                    elif q1 < q2:
                        return 1
                    else:
                        # break ties randomly
                        return random.choice([-1, 1])
                else:
                    # break ties randomly
                    return random.choice([-1, 1])

    @property
    def id(self):
        return self._id

    @property
    def parent(self):
        return self._parent_node

    @property
    def children(self):
        return self._children

    @property
    def branched_children(self):
        return [k for k, v in self._children.items() if v is not None]

    @property
    def is_leaf(self):
        return all(map(lambda x: x is None, self._children.values()))

    @property
    def is_fully_expanded(self):
        return len(self._available_actions) == 0

    @property
    def available_actions(self):
        return self._available_actions

    @property
    def is_root(self):
        return self._parent_node is None

    @property
    def score(self):
        return self._score

    @property
    def visits(self):
        return self._visits

    @property
    def is_terminal(self):
        return self._game_data['done']

    @property
    def game_reward(self):
        if self.is_root and self._game_data['reward'] is None:
            return 0
        return self._game_data['reward']

    @property
    def game_state(self):
        return self._game_data['state']

    @property
    def time(self):
        return self._game_data['t']

    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, action):
        self._action = action

    @property
    def player(self):
        return self._game_data['player']

    @property
    def current_player(self):
        return self._game_data['current_player']

    @property
    def color(self):
        # TODO
        return self._game_data['current_player']

    @property
    def value(self):
        visits = self.visits
        if self.is_root:
            visits -= 1  # the root is visited once at the beginning, but this visit doesn't correspond to any actual game simulation
        if visits == 0:
            return 0
        return self.score / visits

    @property
    def value_variance(self):
        if self.visits == 0:
            return 1

        # sample variance
        # X: r.v., x: realization
        # var(X) = \frac{\sum_i x^2 - n^{-1}\cdot(\sum_i x)^2}{n-1}
        return (self._score_squared - (self._score**2 / self.visits)) / (self.visits - 1)

    @property
    def is_choice(self):
        return False

    @property
    def is_chance(self):
        return False

    @property
    def is_confident(self):
        return self._is_confident

    @is_confident.setter
    def is_confident(self, value):
        assert value is False  # this is only used to invalidate nodes, but a node is confident by default
        self._is_confident = value

    def confidence_interval(self, z, width=True):
        """
        Returns the confidence interval (width) for the value of this node.
        """
        standard_error = np.sqrt((self.value * (1-self.value)) / self.visits)
        half_width = z * standard_error
        if width:
            return 2 * half_width
        else:
            return self.value - half_width, self.value + half_width

    def need_to_expand(self, action):
        return self._children[action] is None

    @staticmethod
    def construct_dummy_node(score, visits):
        node = Node(None, None, [], None, None)
        node.visit(visits)
        node.update_score(score)
        return node

class Tree:

    @staticmethod
    def create_root(root_legal_actions, root_data):
        root = Node(None, 0, root_legal_actions, root_data, None)
        root.visit()
        root.occupancy_frequency = 1
        root.probability = 1
        return root

    def __init__(self, root_legal_actions, root_data, adversarial=False):
        # !!! do not change self to Tree in the next line
        self._root = self.create_root(root_legal_actions, root_data)
        self._nodes = {0: self._root}
        self._last_id = 0
        self._adversarial = adversarial

    def __repr__(self):
        s = ', '.join(map(str, self._nodes))
        return f"Tree({s})"

    def __getitem__(self, index):
        return self._nodes[index]

    def __len__(self):
        return len(self._nodes)

    def insert_node(self, parent_id, action, legal_actions, node_data, **kwargs):
        parent = self._nodes[parent_id]
        new_id = self._last_id + 1
        self._last_id = new_id
        new_node = Node(parent, new_id, legal_actions, node_data, action)
        parent.add_child(new_node)
        self._nodes[new_id] = new_node
        return new_node

    def delete_subtree(self, node, parent):
        """
        This method deletes a subtree that starts from `node` (included). It is only used within the method
        `keep_subtree`.
        """
        self._delete_subtree(node)
        assert node in parent.children.values()
        del parent.children[node.action]
        del self._nodes[node.id]

    def _delete_subtree(self, node):
        """
        This method recursively delete a subtree that starts from `node` (excluded)
        """
        if node.is_leaf:
            return
        for child_id in list(node.children):
            n = node.children[child_id]
            if n is None:
                continue
            self._delete_subtree(n)
            del node.children[child_id]
            try:
                del self._nodes[n.id]
            except KeyError:
                # TODO: this node was deleted following a different subtree
                pass

    def keep_subtree(self, node):
        assert node in self._root.children.values()

        if node is not None:
            # Delete the subtree relative to all the other children
            for action in list(self._root.children):
                n = self._root.children[action]
                if n is node or n is None:
                    continue

                self.delete_subtree(n, self._root)

            # At this point, I still have the root with only the selected child (`node`)
            del self._nodes[self._root.id]
            del self._root
            self._root = node
            self._root.set_root()
        else:
            # TODO: how do you keep a subtree when there is no subtree?
            pass

        assert self._root is self._nodes[self._root.id]


    def visualize(self, node_id=None, level=None, mode='human'):
        if level is None:
            level = self._last_id
        if node_id is None:
            node = self._root
        else:
            node = self._nodes[node_id]
        self._visualize(node, 0, level, mode)

    def _visualize(self, node, depth, level, mode):
        """
        Recursively prints the structure of the tree starting from the given node.

        :param node: The starting node for printing (usually the root node).
        :param depth: The current depth of the node, used for indentation.
        """
        indent = "  " * depth

        if mode == 'human':
            tv = InteractiveTreeVisualizer(self, level, node=node)
            tv.visualize()
            return
        elif mode == 'repr':
            s = node.__repr__()
        else:
            s = node.__str__()

        print(f"{indent}{s}")

        for action, child in node.children.items():
            if child is not None and level > 0:
                self._visualize(child, depth + 1, level-1, mode)

    @property
    def root(self):
        return self._root

    @property
    def nodes(self):
        return self._nodes

    @property
    def adversarial(self):
        return self._adversarial

    def principal_variation(self, action, action_decoder_fn=None):
        pv = []
        parent = self.root
        child = parent.children[action]
        while True:
            pv.append((parent.current_player, child.action))
            if child.is_leaf:
                break
            parent, child = child, child.best_child

        if action_decoder_fn is None:
            return ' -> '.join([str(n[1]) for n in pv])
        else:
            return ' -> '.join([action_decoder_fn(n[1], player=n[0]) for n in pv])


    def backup(self):
        bkp = {
            'root': self._root.id,
            'nodes': self._nodes.copy(),
            'last_id': self._last_id,
        }
        return bkp

    def load(self, backup):
        self._nodes = backup['nodes']
        self._last_id = backup['last_id']
        self._root = self.nodes[backup['root']]
        assert self._root.is_root
        assert self._root.id == 0
        assert len(self._nodes) == self._last_id + 1
