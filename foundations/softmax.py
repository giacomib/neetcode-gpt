import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max = np.max(z)

        for count in range (len(z)):
            z[count] -= max
        
        sum = np.sum(np.exp(z))

        res = np.array(z)

        for count in range (len(z)):
            res[count] = (pow(np.e, z[count]) / sum)
        return np.round(res, 4)
