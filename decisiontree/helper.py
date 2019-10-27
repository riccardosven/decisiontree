"Helper definitions for decisiontree"

from collections import Counter
from collections.abc import Iterable
from typing import List

class Datum:
    """Object defining one datum in the problem (a pair of features and class)"""

    def __init__(self, point, label):
        self.point = point
        self.label = label


class DecisionTreeNode:
    """Node inside the decision tree

      A feature of the input point is checked against a threshold, if it is less we follow the left branch,
      else we follow the right branch.
    """

    def __init__(self, feature, threshold, left=None, right=None):
        self.feature = feature  # Index of the feature this node checks
        self.threshold = threshold  # Threshold: if less than this, go left, else go right
        self.left = left
        self.right = right

    def predict(self, point):
        """Predict the class of the point given in input"""
        if isinstance(point[0], Iterable):
            return [self.predict(p) for p in point]
        else:
            if point[self.feature] < self.threshold:
                return self.left.predict(point)
            return self.right.predict(point)

    def __str__(self):
        return f"( {self.feature} < {self.threshold} ? {str(self.left)} : {str(self.right)}))"


class DecisionTreeLeaf:
    "Leaf on the decision tree"

    def __init__(self, prediction):
        self.prediction = prediction  # Prediction returned by this leaf node

    def predict(self, _):
        """Predict the class of the point given in input (this is a leaf node, so the class is decided and returned)"""
        return self.prediction

    def __str__(self):
        return f"( {self.prediction} )"


def gini_impurity(data: List[Datum]):
    """Returns the Gini impurity of the data (is zero when all the labels are the same)"""
    ndata = len(data)
    counts = Counter()
    for datum in data:
        counts[datum.label] += 1

    impurity = 1
    for c in counts.values():
        impurity -= (c/ndata)**2
    return impurity  # Gini impurity


def binary_impurity(data: List[Datum]):
    """Impurity measure for binary classification"""
    number_ones = 0
    for datum in data:
        number_ones += datum.label
    return min(number_ones/len(data), 1-number_ones/len(data))
