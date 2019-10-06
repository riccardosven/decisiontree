"Implements a naive classification tree algorithm for numeric features"

from collections import Counter
from decisiontree.helper import TreeNode, TreeLeaf, gini_impurity


class DecisionTree:
    "Base class for the decision tree"

    def __init__(self, impurity=gini_impurity):
        self.root = None
        self.data = None
        self.targets = None
        self.impurity_measure = impurity

    def train(self, dataset):
        "Train the tree on data"
        self.data = dataset[0]  # Input features
        self.targets = dataset[1]  # Targets
        self.root = self._best_split(
            list(range(len(self.targets))), list(range(len(self.data[0]))))

    def predict(self, points):
        "Predict the class labels of a set of points"
        return [self.root.predict(point) for point in points]

    def _find_split(self, indices, cutoff, feature):
        "Find the split among the indices given a cutoff value and a feature"
        left = []  # Indices where feature is lower than cutoff
        right = []  # Indices where feature is larger than target
        for idx in indices:
            point = self.data[idx]
            if point[feature] <= cutoff:
                left.append(idx)
            else:
                right.append(idx)

        impurity_left = self.impurity_measure(self.targets[left])
        impurity_right = self.impurity_measure(self.targets[right])
        impurity = (len(left) * impurity_left + len(right)
                    * impurity_right)/len(indices)

        return impurity, left, right

    def _best_split(self, indices, features):
        "Find the best splitting point of the indices among the features"
        root_impurity = self.impurity_measure(self.targets[indices])
        best_impurity = root_impurity
        best_rule = (0, 0)
        for feature in features:
            for idx in indices:
                cutoff = self.data[idx][feature]
                impurity, left, right = self._find_split(
                    indices, cutoff, feature)
                if impurity <= best_impurity:
                    best_impurity = impurity
                    best_rule = (feature, cutoff)
                    best_left = left
                    best_right = right

        if best_impurity >= root_impurity:
            return TreeLeaf(Counter(self.targets[indices]).most_common(1)[0][0])
        else:
            features.remove(best_rule[0])
            left = self._best_split(best_left, features)
            right = self._best_split(best_right, features)
            return TreeNode(best_rule, left, right)

    def __str__(self):
        return str(self.root)
