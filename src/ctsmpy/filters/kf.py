from ctsmpy.filters import BaseFilter as BaseFilter
from typing import Union, Optional
import numpy as np
import pandas as pd
from numpy.typing import NDArray


class LTIKalmanFilter(BaseFilter):
    '''
    Linear Time-Inveriant Kalman Filter class.

    '''
    def __init__(self, A: NDArray[np.float32], 
                       B: NDArray[np.float32], 
                       C: NDArray[np.float32], 
                       D: NDArray[np.float32], 
                       S: NDArray[np.float32]
                       ) -> None:

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.S = S
        k = A.shape[0]
        super().__init__(k)

    def _time_update(self,
                     x: NDArray[np.float32],
                     u: NDArray[np.float32],
                     P: NDArray[np.float32],
                     i: int,
                     ) -> Union[NDArray[np.float32],
                                NDArray[np.float32]]:
        if u is None:
            xp = self.A @ x
        else:
            xp = self.A @ x + self.B @ u

        Pp = self.A @ P + P @ self.A.T + self.S @ self.S.T
      
        return xp, Pp

    def _data_update(self,
                     x: NDArray[np.float32],
                     P: NDArray[np.float32],
                     e: NDArray[np.float32],
                     K: NDArray[np.float32],
                     R: NDArray[np.float32],
                     i: int,
                     ) -> Union[NDArray[np.float32],
                                NDArray[np.float32]]:
        
        x = x + K @ e
        P = P - K @ R @ K.T

        return x, P

    
    def _output_prediction(self,
                    x: NDArray[np.float32],
                    P: NDArray[np.float32],
                    u: NDArray[np.float32],
                    i: int,
                    ) -> Union[NDArray[np.float32],
                                NDArray[np.float32],
                                NDArray[np.float32]]:
        if u is None:
            y = self.C @ x
        else:
            y = self.C @ x + self.D @ u
        R = self.C @ P @ self.C.T + self.S

        K = P @ self.C.T @ np.linalg.inv(R)
        
        return y, R, K



if __name__ == '__main__':
    from time import time


    N = 100
    M = 2
    A = np.random.randn(M, M) * 0.001
    A = A @ A.T
    B = np.random.randn(M, 1)
    C = np.random.randn(1, M)
    D = np.random.randn(1, 1)

    S = np.array([[0.01]])

    u = np.random.randn(N, 1)
    x = np.zeros((N, M))
    y = np.zeros((N, 1))
    x[0] = np.array([0.5]*M)

    for i in range(1, N):
        x[i] = A @ x[i-1] + B @ u[i-1]
        y[i] = C @ x[i] + D @ u[i]

    kf = LTIKalmanFilter(A, B, C, D, S)
    x0 = np.zeros(M)
    P0 = np.eye(M)

    t0 = time()
    kf.predict(x0, P0, u, y)
    t1 = time()
    print(t1-t0)
    
    print(kf.xp)

    print(kf.e)

