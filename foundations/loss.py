import numpy as np
from numpy.typing import NDArray

class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        sum = 0
        for index in range(0, len(y_pred)):
            sum = sum + ( y_true[index] * np.emath.log(y_pred[index]) + ( (1 - y_true[index]) * np.emath.log(1 - y_pred[index])) ) 

        loss = ( -1 / len(y_pred) ) * sum
        return np.round(loss, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        for index in range(0, len(y_pred)):
            y_pred[index] = y_pred[index] + 1e-7

        
        #first_sum = 0
        #    first_sum = first_sum + index
        #for index in range(0, len(y_pred)):

        #second_sum = 0
        #for c in range(0, len(y_pred)):
        #    second_sum = second_sum + ( y_true[c] * np.emath.log(y_pred[c]) )

        sum = 0
        for i in range (0, len(y_true)):
            for c in range (0, len(y_pred)):
                sum = sum + y_true[i, c] * np.log(y_pred[i, c])

        loss = ( -1 / len(y_pred) ) * sum
        return np.round(loss, 4)
