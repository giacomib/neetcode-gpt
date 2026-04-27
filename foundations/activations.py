import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        for count in range(len(z)):
            z[count] = 1 / (1 + np.pow(np.e, -z[count]))
        return np.round(z, 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        for count in range(len(z)):
            if z[count] < 0:
                z[count] = 0
        return z
