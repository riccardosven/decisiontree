"Helper definitions for decisiontree"

from collections import Counter
from numpy import log


class TreeNode:
    "Node inside the tree"

    def __init__(self, rule, left, right):
        self.rule = rule  # Decision rule
        self.left = left  # If rule is true
        self.right = right  # If rule is false

    def predict(self, point):
        "Predict the class of a new input"
        if point[self.rule[0]] <= self.rule[1]:
            return self.left.predict(point)
        else:
            return self.right.predict(point)

    def __str__(self):
        return f"({self.rule[0]} <= {self.rule[1]} ? {str(self.left)} : {str(self.right)})"


class TreeLeaf:
    "Leaf node of the tree"

    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, _):
        "Return the leaf prediction"
        return self.prediction

    def __str__(self):
        return f"({self.prediction})"


def gini_impurity(classes):
    "Returns the Gini impurity of a vector of classes"
    counts = Counter(classes)
    return 1 - sum([(c/len(classes))**2 for c in counts.values()])


def entropy(classes):
    "Returns the entropy of a vector of classes"
    counts = Counter(classes)
    return -sum([(c/len(classes)*log(c/len(classes))) for c in counts.values()])
