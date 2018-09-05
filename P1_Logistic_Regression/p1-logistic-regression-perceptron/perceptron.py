import numpy as np
class Perceptron:
    """
    Class to represent a logistic regression model.
    """

    def __init__(self, l_rate, epochs, n_features):
        """
        Create a new model with certain parameters.

        :param l_rate: Initial learning rate for model.
        :param epoch: Number of epochs to train for.
        :param n_features: Number of features.
        """
        self.l_rate = l_rate
        self.epochs = epochs
        self.coef = [0.0] * n_features
        self.bias = 0.0

    def predict(self, features):
        """
        Given an example's features and the coefficients, predicts the class.

        :param features: List of real valued features for a single training example.

        :return: Returns the predicted class (either 0 or 1).
        """
        



        return 1.0 if np.sum(np.convolve(np.array(self.coef),np.array(features)))+self.bias>0.0 else 0.0

    def sg_update(self, features, label):
        """
        Computes the update to the weights based on a predicted example.

        :param features: Features to train on.
        :param label: Corresponding label for features.
        """
        yhat = self.predict(features)
        error=label-yhat
        self.coef=np.array(self.coef)+self.l_rate*error*np.array(features)
        self.bias=np.array(self.bias)+self.l_rate*error



        return

    def train(self, X, y):
        """
        Trains the model on training data.

        :param X: Features to train on.
        :param y: Corresponding label for each set of features.
        """
        for epoch in range(self.epochs):
            for features, label in zip(X, y):
                self.sg_update(features, label)
        return self.bias, self.coef
