import numpy as np
from scipy.special import softmax

from src.tree.tree import Tree, Node

def ucb(node, parent, c=np.sqrt(2)):
    """
    Calculates the Upper Confidence Bound for a tree.
    :param node: the node for which it calculates the UCB
    :param parent: the parent node of `node`
    :param c: the coefficient of the formula
    """

    exploitation = node.score / node.visits
    if parent.visits == 0:
        exploration = 0
    else:
        exploration = np.sqrt(
            np.log(parent.visits) / node.visits
        )
    return exploitation + c * exploration

class ChanceNode(Node):
    """
        Analogies with superclass:
        - always has only one parent
        Differences from superclass:
        - doesn't keep statistics
        - doesn't use UCT formula during selection

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self._game_data

    def __repr__(self):
        return f"Chance(id={self.id}, visits={self.visits}, score={self.score}, action={self._action}, prob={round(self._probability, 3)}, 𝝺={self.occupancy_frequency})"

    def __str__(self):
        return f"Chance(id={self.id}, visits={self.visits}, score=n.a., action={self._action})"

    @property
    def features(self):
        raise RuntimeError

    @features.setter
    def features(self, new_features):
        raise RuntimeError

    @property
    def score2(self):
        return sum(map(lambda n: n.score * n.features['value'], filter(lambda n: n is not None, self.children.values()))) / self.visits

    @property
    def is_chance(self):
        return True

    def update_distribution(self):
        """
        ChanceNodes have random transition, hence the count of the visits of the children really reflects the
        probability of visiting them
        """
        return super().update_distribution()


class ChoiceNode(Node):
    """
        Analogies with superclass:
        - keeps the statistics
        - uses UCT formula during selection
        Differences from superclass:
        - has multiple parents (not for now, need to implement state hashing)

    """
    def __init__(self, parent_node, *args, **kwargs):
        super().__init__(parent_node, *args, **kwargs)
        del self._parent_node
        self._parent_nodes = {parent_node.id: parent_node} if parent_node is not None else {}

    def __repr__(self):
        return f"Choice(id={self.id}, visits={self.visits}, score={self.score}, state={self._action}, #parents={len(self._parent_nodes)}, prob={round(self._probability, 3)}, 𝝺={self.occupancy_frequency})"

    def __str__(self):
        return f"Choice(id={self.id}, visits={self.visits}, score=n.a., state={self._action}, #parents={len(self._parent_nodes)})"

    def set_root(self):
        assert self._parent_nodes is not None
        assert not all(map(lambda n: n is None, self._parent_nodes.values()))
        self._parent_nodes = {}
        self._occupancy_frequency = 1
        self._probability = 1

    def add_parent(self, parent):
        self._parent_nodes[parent.id] = parent

    @property
    def parent(self):
        # TODO this is going to be broken with hashing
        assert len(self._parent_nodes) == 1
        return list(self._parent_nodes.values())[0]

    @property
    def parents(self):
        return self._parent_nodes

    @property
    def is_root(self):
        return self._parent_nodes is None or all(map(lambda n: n is None, self._parent_nodes.values()))

    @property
    def is_choice(self):
        return True

    @staticmethod
    def generate_node_hash(node_data):
        state = node_data['state']
        t = node_data['t']
        return state, t

    def update_distribution(self):
        """
        ChoiceNodes choose their children based on the UCT formula
        """
        return super().update_distribution()
        if self.is_leaf:
            return

        children_ucbs = [(ch, ucb(n, self, 5)) for ch, n in self.children.items() if n is not None]
        sorted_children = sorted(children_ucbs, key=lambda c: c[0])

        keys, values = zip(*sorted_children)

        # apply softmax
        softmax_values = softmax(values)

        softmax_dict = dict(zip(keys, softmax_values))

        self._distribution = {ch: softmax_dict[ch] for ch, n in self.children.items() if n is not None}
        for ch, n in self._children.items():
            if n is not None:
                n.probability = self._distribution[ch]


class ChanceTree(Tree):
    """
        This class is a special case of a tree, used only for non-adversarial, single-agent settings. In this
        implementation, two things are mandatory:
        - nodes must be alternating between Chance and Choice
        - an episode must necessarily start with a Choice node and end with a Chance node
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._choice_nodes = dict()

    @staticmethod
    def create_root(root_legal_actions, root_data):
        root = ChoiceNode(None, 0, root_legal_actions, root_data, None)
        root.visit()
        root.occupancy_frequency = 1
        root.probability = 1
        return root

    @property
    def choice_nodes(self):
        return list(filter(lambda n: n is not None and isinstance(n, ChoiceNode), self.nodes.values()))

    def insert_node(self, parent_id, action, legal_actions, node_data, chance=None):
        assert chance is not None

        parent = self._nodes[parent_id]
        new_id = self._last_id + 1
        self._last_id = new_id

        if chance:
            new_node = ChanceNode(parent, new_id, legal_actions, node_data, action)
            parent.add_child(new_node)
            self._nodes[new_id] = new_node
        else:
            new_node = ChoiceNode(parent, new_id, legal_actions, node_data, action)
            parent.add_child(new_node)
            self._nodes[new_id] = new_node
            node_hash = ChoiceNode.generate_node_hash(node_data)
            self._choice_nodes[node_hash] = new_node

        return new_node

    def get_choice_node_if_existing(self, node_hash):
        return None
        # return self._choice_nodes.get(node_hash)

    def delete_choice_node(self, node_hash):
        # del self._choice_nodes[node_hash]
        pass

    def delete_subtree(self, node, parent):
        """
        This method deletes a subtree that starts from `node` (included). It is only used within the method
        `keep_subtree`.
        """
        self._delete_subtree(node)
        assert node in parent.children.values()
        parent.children[node.action] = None
        parent.available_actions.append(node.action)
        del self._nodes[node.id]

    def _delete_subtree(self, node):
        """
        This method recursively delete a subtree that starts from `node` (excluded)
        """
        if node.is_leaf:
            return
        for child_id in node.children:
            child_node = node.children[child_id]
            if child_node is None:
                continue
            self._delete_subtree(child_node)

            if isinstance(child_node, ChoiceNode):
                assert node.id in child_node.parents
                child_node.parents.pop(node.id)
                if len(child_node.parents) == 0:
                    del self._nodes[child_node.id]
                    s = child_node.game_state
                    t = child_node.time
                    self.delete_choice_node((s,t))
            else:
                del self._nodes[child_node.id]

            node.children[child_id] = None
            node.available_actions.append(child_node.action)
