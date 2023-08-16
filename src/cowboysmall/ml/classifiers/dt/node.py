from collections import Counter


class Node:

    def __init__(self, criteria, left, right):
        self.criteria = criteria
        self.left     = left
        self.right    = right

    def predict(self, row):
        if isinstance(row[self.criteria[0]], int) or isinstance(row[self.criteria[0]], float):
            return self.left.predict(row) if row[self.criteria[0]] <  self.criteria[1] else self.right.predict(row)
        else:
            return self.left.predict(row) if row[self.criteria[0]] == self.criteria[1] else self.right.predict(row)

    def to_string(self, depth = 1):

        return '{}{} < {:8.5f}\n{}\n{}'.format(
            depth * ' ',
            self.criteria[0],
            self.criteria[1],
            self.left.to_string(depth + 1),
            self.right.to_string(depth + 1)
        )


class LeafNode:

    def __init__(self, counts):
        self.counts = Counter(counts)

    def predict(self, row):
        return self.counts.most_common()[0][0]

    def to_string(self, depth = 1):
        return '{}{}'.format(depth * ' ', self.counts)
