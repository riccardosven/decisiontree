from typing import List
from decisiontree.helper import Datum, DecisionTreeNode, DecisionTreeLeaf, gini_impurity


class DecisionTree:
    def __init__(self, impurity_measure=gini_impurity):
        self.tree = None
        self._impurity = impurity_measure

    def fit(self, data: List[List[float]], labels: List[int]):
        """Fit the decision tree based on data and labels"""
        dataset = [Datum(point, label) for point, label in zip(data, labels)]
        # Features available for the trainer
        features = list(range(len(data[0])))
        self.tree = self._train(dataset, features)

    def _train(self, data: List[Datum], features: List[int]):
        """Recursive training function using `data` as input. It is allowed to use the `features` to find an optimal split"""
        ndata = len(data)

        if self._impurity(data) == 0:
            # If the impurity is zero, return a Leafnode with the correct label
            return DecisionTreeLeaf(data[0].label)

        best_impurity = float("inf")
        for feature in features:
            # Sort the data according to the values of the column `feature`
            data.sort(key=lambda x: x.point[feature]
                      )  # pylint: disable=cell-var-from-loop
            for i in range(0, len(data) - 1):
                # We split the data at the midpoint between this
                cutoff = 0.5 * \
                    (data[i + 1].point[feature] + data[i].point[feature])
                # feature and the next, this means that we never
                # have empty splits to take care of
                left_impurity = self._impurity(data[0: i + 1])
                right_impurity = self._impurity(data[i + 1:])
                average_impurity = (
                    left_impurity * i + right_impurity * (ndata - i)
                ) / ndata

                if average_impurity < best_impurity:
                    best_impurity = average_impurity
                    best_feature = feature
                    best_split = i
                    best_cutoff = cutoff

        # Sort the data according to the chosen feature to ease splitting
        data.sort(key=lambda x: x.point[best_feature])
        node = DecisionTreeNode(best_feature, best_cutoff)  # New node
        # Recursively train on the splitted dataset
        node.left = self._train(data[0: best_split + 1], features)
        node.right = self._train(data[best_split + 1:], features)

        return node

    def predict(self, sample: List[float]) -> int:
        """Predicts the class of the sample in input"""
        return self.tree.predict(sample)
