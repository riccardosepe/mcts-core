from libs.mcts.src.ai import MCTS
from libs.mcts.src.tree import ChanceTree, ChoiceNode, ChanceNode


class ChanceMCTS(MCTS):
    """
    Differences from superclass:
    - the Tree is necessarily a ChanceTree
    - the selection depends on the nature of the Node
    - the backpropagation skips the chance nodes
    """

    def __init__(self, *args, **kwargs):
        kwargs['adversarial'] = False
        super().__init__(*args, **kwargs)
        self.trajectory = None

    def _build_tree(self):
        tree = ChanceTree(self.transition_model.legal_actions, self.transition_model.backup())
        if self._evaluator is not None:
            root_state = self.transition_model.s
            root_time = self.transition_model.t
            root_features = self._evaluator.evaluate(root_state, root_time)
            tree.root.features = root_features

        return tree

    def _select(self):
        node = self.tree.root
        self.trajectory = [node]
        while not node.is_leaf:
            chance_node = self.select_ucb(node)
            if chance_node is self.DUMMY_NODE:
                return node, chance_node.action
            s, _, _, _, _ = self.transition_model.step(chance_node.action)
            # TODO: HASHING. FOR THE MOMENT (FROZEN LAKE) THE STATE IS JUST AN INTEGER
            self.t += 1

            self.trajectory.append(chance_node)

            if chance_node.children[s] is not None:
                node = chance_node.children[s]
                self.trajectory.append(node)
            else:
                # If the trajectory ends with a chance node, it's the chance node that has to be expanded
                return chance_node, None
        return node, None

    def _expand(self, node, action=None):
        if isinstance(node, ChoiceNode):
            if action is None:
                action = node.random_action()
            support_random_action = self.transition_model.next_states(action)
            s, _, _, _, _ = self.transition_model.step(action)

            # insert first a chance node
            new_chance_node = self.tree.insert_node(node.id,
                                                    action=action,
                                                    legal_actions=support_random_action,
                                                    node_data=None,
                                                    chance=True)
            self.trajectory.append(new_chance_node)
        elif isinstance(node, ChanceNode):
            new_chance_node = node
        else:
            raise RuntimeError

        # then insert a choice node if it's not hashed
        new_choice_node = self._insert_or_get_choice_node(new_chance_node, self.transition_model.backup())
        self.t += 1

        self.trajectory.append(new_choice_node)
        return new_choice_node

    def _evaluate(self, leaf_node):
        if self._evaluator is None:
            return super()._rand_rollout_eval(leaf_node)
        else:
            node_state = leaf_node.game_state
            node_time = leaf_node.time
            features =  self._evaluator.evaluate(node_state,
                                                        node_time,
                                                        # NB: parent is a ChanceNode, we want the parent ChoiceNode
                                                        parent_features=leaf_node.parent.parent.features)

            leaf_node.features = features
            value = features['value']
            if leaf_node.is_terminal:
                # NB: in theory this line is useless as the heuristic value function interpolates the reward function
                value = leaf_node.game_reward
            return value

    def _backpropagate(self, _, score, visits=1):
        assert len(self.trajectory) > 0
        assert isinstance(self.trajectory[-1], ChoiceNode)
        while len(self.trajectory) > 0:
            node = self.trajectory.pop()
            node.update_score(score)
            node.visit(visits)
            node.update_distribution()

    def _backpropagate_parents(self, node, score, visits=1):
        if node is None:
            return

        if isinstance(node, ChoiceNode):
            node.update_score(score)

        node.visit(visits)  # count the visits also for the chance nodes

        if isinstance(node, ChanceNode):
            self._backpropagate(node.parent, score * self.gamma, visits)
        else:
            for parent in node.parents.values():
                self._backpropagate(parent, score * self.gamma, visits)

    def determinize_chance_node(self, state):
        new_root = self.tree.root.children[state]
        self.tree.keep_subtree(new_root)

    def step_consistency(self, action=None):
        # We are in a chance tree -> if we want to determinize, we're actually
        # determinizing an action
        state = action
        if self._keep_subtree:
            assert state is not None
            self.determinize_chance_node(state)
        else:
            self.reset()

    def _insert_or_get_choice_node(self, parent_node, node_data):
        s = node_data['state']
        node_data = self.transition_model.backup()
        node_hash = ChoiceNode.generate_node_hash(node_data)
        hashed_node = self.tree.get_choice_node_if_existing(node_hash)
        if hashed_node is None:
            new_choice_node = self.tree.insert_node(parent_node.id,
                                                    action=s,
                                                    legal_actions=self.transition_model.legal_actions,
                                                    node_data=self.transition_model.backup(),
                                                    chance=False)
        else:
            new_choice_node = hashed_node
            # IMPORTANT: when a hashed node is detected, it's important to first backpropagate its statistics to the root
            # and then go on with the simulation. At that point, backpropagate the new results along all paths
            self._backpropagate_parents(parent_node, hashed_node.score, hashed_node.visits)

            parent_node.add_child(hashed_node)
            hashed_node.add_parent(parent_node)

        return new_choice_node