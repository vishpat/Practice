#!/usr/bin/python


class node(object):

    def __init__(self, value):
        self._value = value
        self._left = None
        self._right = None
        self._height = 0

    def add_child(self, n):

        if n < self._value:
            if self._left is None:
                self._left = node(n)
            else:
                self._left.add_child(n)
        else:
            if self._right is None:
                self._right = node(n)
            else:
                self._right.add_child(n)

    def dfs_visit(self, f):
        if self._left is not None:
            self._left.dfs_visit(f)
        if self._right is not None:
            self._right.dfs_visit(f)

        f(self._value)

    def height(self):
        left_height = 0 if self._left is None else 1 + self._left.height()
        right_height = 0 if self._right is None else 1 + self._right.height()
        return max(left_height, right_height)

    def is_balanced(self):
        left_height = 0 if self._left is None else 1 + self._left.height()
        right_height = 0 if self._right is None else 1 + self._right.height()
        return left_height == right_height

    # Least Common Ancestor
    def lca(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        p_subtree = self.get_subtree(root, p)
        q_subtree = self.get_subtree(root, q)

        i = 0
        while (i < len(p_subtree) and i < len(q_subtree)
               and p_subtree[i] == q_subtree[i]):
            i += 1

        return p_subtree[i - 1]

    def get_subtree(self, root, p):

        if root._value == p:
            return []

        if root._left is None and root._right is None:
            return None

        if root._left:
            subtree = self.get_subtree(root._left, p)
            if subtree is not None:
                subtree.insert(0, root._value)
                return subtree

        if root._right:
            subtree = self.get_subtree(root._right, p)
            if subtree is not None:
                subtree.insert(0, root._value)
                return subtree

        return None


def print_func(x):
    print(x)


if __name__ == "__main__":
    root = node(10)
    for e in [4, 20, 2, 6, 16, 22, 12, 18]:
        root.add_child(e)

    print(root.lca(root, 12, 6))
