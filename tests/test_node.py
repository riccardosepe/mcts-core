import unittest

from src.tree import Node
from unittest import TestCase



class TestNode(TestCase):

    def test_add_child(self):
        root = Node(None, 0, [1, 2], {'name': 'root'}, None)
        node = Node(None, 1, [2], {'name': 'node'}, 1)

        root.add_child(node)

        assert root.children[1] is node


    def test_visit(self):
        node = Node(None, 1, [2], {'name': 'node'}, 1)

        visits = node.visits
        node.visit()

        assert node.visits == visits + 1


    def test_update_score(self):
        node = Node(None, 1, [2], {'name': 'node', 'player': -1}, 1)

        score = node.score
        node.update_score(1)
        assert node.score == score + 1


    def test_random_action(self):
        node = Node(None, 1, [0, 2, 3], {'name': 'node'}, 1)

        a1 = node.random_action()
        assert a1 in node.available_actions


    def test_set_root(self):
        root = Node(None, 0, [1, 2], {'name': 'root'}, None)
        node = Node(root, 1, [2], {'name': 'node'}, 1)

        root.add_child(node)

        assert node.parent is not None
        node.set_root()
        assert node.parent is None

    def test_ply(self):
        root = Node(None, 0, [1, 2], {'name': 'root'}, None)

        root.ply(1)

        assert 1 not in root.available_actions
        assert 1 not in root.children

    def test_best_child(self):
        root = Node(None, 0, [1, 2, 3], {'name': 'root'}, None)
        child1 = Node(root, 1, [2, 3], {'name': 'child1'}, 1)
        child2 = Node(root, 2, [1, 3], {'name': 'child2'}, 2)
        child3 = Node(root, 3, [1, 2], {'name': 'child3'}, 3)

        root.add_child(child1)
        root.add_child(child2)
        root.add_child(child3)

        child1._visits = 5
        child2._visits = 5
        child3._visits = 4

        child1._score = 5
        child2._score = 4
        child3._score = 3

        assert root.best_child is child1

        child1._visits = 4

        assert root.best_child is child2


    def test_is_leaf(self):
        root = Node(None, 0, [1, 2], {'name': 'root'}, None)
        assert root.is_leaf


    def test_is_fully_expanded(self):
        root = Node(None, 0, [1, 2, 3], {'name': 'root'}, None)
        child1 = Node(root, 1, [2, 3], {'name': 'child1'}, 1)
        child2 = Node(root, 2, [1, 3], {'name': 'child2'}, 2)
        child3 = Node(root, 3, [1, 2], {'name': 'child3'}, 3)

        root.add_child(child1)
        root.add_child(child2)
        root.add_child(child3)

        assert root.is_fully_expanded


if __name__ == '__main__':
    unittest.main()