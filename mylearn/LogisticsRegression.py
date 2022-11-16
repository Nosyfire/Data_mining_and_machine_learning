import numpy as np


def sigmoid(X: np.array):
    return 1 / (1 + np.exp(-X))


def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)


class LogisticRegression():

    def __init__(self, lr: float = 0.001, n_iters: int = 1000):
        """
        :param lr: learning rate
        :param n_iters: number of iteration
        """
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X: np.array, y: np.array):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict_proba(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        self.probability = y_pred
        return np.array(y_pred)

    def predict(self, X):
        class_pred = [0 if y <= 0.5 else 1 for y in self.predict_proba(X)]
        return np.array(class_pred)



