"""
Disjoint Set (Union-Find) data structure.

Ported from src/lib/disjoint_set.cpp — used for merging nearby contours.
Uses union-by-rank and path compression for near O(1) amortized operations.
"""


class DisjointSets:
    """Union-Find data structure for grouping elements into disjoint sets."""

    def __init__(self, n: int):
        """Initialize n singleton sets {0}, {1}, ..., {n-1}."""
        self.parent = list(range(n))
        self.rank = [0] * n
        self.n = n

    def find(self, x: int) -> int:
        """Find the representative (root) of element x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int):
        """Merge the sets containing x and y using union-by-rank."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        # Attach the smaller tree under the root of the larger tree.
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
