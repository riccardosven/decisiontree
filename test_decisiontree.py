"Testing the decision tree"
import numpy as np
from sklearn.datasets import load_wine, load_iris
from decisiontree.core import DecisionTree


def test_dataset(dataset, niter=20):
    "Takes in a dataset in sklearn format and computes the average accuracy over niter trials"
    accuracy = []

    for _ in range(niter):
        idx = np.random.permutation(len(dataset.target))
        ntrain = int(0.7*len(idx))
        train = idx[:ntrain]
        test = idx[ntrain:]
        tree = DecisionTree()
        tree.train((dataset.data[train, :], dataset.target[train]))

        predictions = tree.predict(
            dataset.data[test, :]) == dataset.target[test]
        accuracy.append(np.mean(predictions))

    return np.mean(accuracy)


def main():
    "Test decision tree on IRIS and WINE datasets"
    accuracy_iris = test_dataset(load_iris())
    accuracy_wine = test_dataset(load_wine())

    print(f"Average accuracy for IRIS: {accuracy_iris:.3f}")
    print(f"Average accuracy for WINE: {accuracy_wine:.3f}")


if __name__ == "__main__":
    main()
