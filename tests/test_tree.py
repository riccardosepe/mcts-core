import unittest

from unittest import TestCase

from ..src.tree import Tree


class TestTree(TestCase):

    def test_insert_node(self):
        tree = Tree([42], {'name': 'root'})
        child = tree.insert_node(0, 42, [7], {'name': 'child'})
        root = tree.root

        assert child.parent is root
        assert len(tree._nodes) == 2
        assert child in root.children.values()
        # TODO: more conditions?



    def test_delete_subtree(self):
        tree = Tree([1, 2], {'name': 'root'})
        _ = tree.insert_node(0, 1, [7], {'name': 'child1'})
        child2 = tree.insert_node(0, 2, [9], {'name': 'child2'})
        _ = tree.insert_node(child2.id, 9, [4], {'name': 'child3'})

        tree.delete_subtree(child2)
        assert len(tree._nodes) == 2


    def test_keep_subtree(self):
        tree = Tree([1, 2], {'name': 'root'})
        child1 = tree.insert_node(0, 1, [7], {'name': 'child1'})
        child2 = tree.insert_node(0, 2, [9], {'name': 'child2'})
        _ = tree.insert_node(child2.id, 9, [4], {'name': 'child3'})

        tree.keep_subtree(child1)

        assert len(tree._nodes) == 1
        assert child1 in tree._nodes.values()
        assert tree.root is child1


if __name__ == '__main__':
    unittest.main()