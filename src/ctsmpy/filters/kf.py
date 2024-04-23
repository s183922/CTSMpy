from ctsmpy.filters import BaseFilter as BaseFilter
from typing import Union, Optional
import numpy as np
import pandas as pd
from numpy.typing import NDArray

class KalmanFilter(BaseFilter):
    def __init__(self, k: int = 1) -> None:
        super().__init__(k)



    def get_system_matrices(self) -> None:
        '''
        Gets the system matrices.
        '''
        pass

    def _predict(self) -> None:
        '''
        Predicts the state of the system.
        '''

        # Predict the state
        self.x = self.F @ self.x
