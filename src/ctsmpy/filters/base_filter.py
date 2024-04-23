import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import Union, Optional


class BaseFilter():
    '''
    Base class for all filters.
    '''
    def __init__(self, k: int = 1) -> None:
        self.data: Optional[NDArray[np.float32]] = None     # Data to be filtered
        self.N: Optional[int] = None                        # Number of samples
        self.M: Optional[int] = None                        # Number of observation
        self.inputs: Optional[NDArray[np.float32]] = None   # Inputs
        self.U: Optional[int] = None                        # Number of inputs
        self.t: Optional[NDArray[np.float32]] = None        # Time vector
        self.k: int = k                                     # Number of states

    def _mountdata(self,
                  data: Union[NDArray[np.float32], pd.Series, pd.DataFrame],
                  inputs: Union[NDArray[np.float32], pd.Series, pd.DataFrame, None] = None,
                  t: Union[NDArray[np.float32], pd.Series, pd.DataFrame, float, int, None] = None,
                    ) -> None:
        '''
        Mounts the data to be filtered.

        Parameters
        ----------
        data : numpy array, pandas Series or pandas DataFrame
            Data to be filtered. If data is 2D, the first dimension must be the number of samples
            and the second dimension must be the number of observation.

        inputs : numpy array, pandas Series or pandas DataFrame, optional
            Inputs. If inputs is 2D, the first dimension must be the number of samples and the
            second dimension must be the number of inputs. The default is None.

        t : numpy array, pandas Series, pandas DataFrame, int, float, optional
            Time vector. If t is a scalar, it will be considered the time step and a time vector
            will be created. The default is None.

        Raises
        ------
        TypeError
            If data, inputs or t are not numpy arrays, pandas Series or pandas DataFrames.
        ValueError
            If the number of samples in data and inputs are not the same.
            If the number of samples in data and t are not the same.
            If t is not 1D.
            If t is not a time vector.
        '''
        # Check types
        if not isinstance(data, (np.ndarray, pd.Series, pd.DataFrame)):
            raise TypeError('data must be a numpy array, pandas Series or pandas DataFrame.')
        if inputs is not None:
            if not isinstance(inputs, (np.ndarray, pd.Series, pd.DataFrame)):
                raise TypeError('inputs must be a numpy array, pandas Series or pandas DataFrame.')
        if t is not None:
            if not isinstance(t, (np.ndarray, pd.Series, pd.DataFrame, int, float)):
                raise TypeError('t must be a numpy array, pandas Series, pandas DataFrame or int.')

        # Convert to numpy array
        if isinstance(data, (pd.Series, pd.DataFrame)):
            data = data.values
        # Get the number of samples and number of observations
        self.data = data

        # Check if the data is 2D
        if len(data.shape) == 1:
            self.N = data.shape[0]
            self.M = 1
        else:
            self.N, self.M = data.shape

        # Convert inputs and t to numpy array
        if inputs is not None:
            if isinstance(inputs, (pd.Series, pd.DataFrame)):
                inputs = inputs.values
            self.inputs = inputs
            self.U = inputs.shape[1] if len(inputs.shape) > 1 else 1
            # Check if the number of samples is the same
            if inputs.shape[0] != self.N:
                raise ValueError('The number of samples in data and inputs must be the same.')

        if t is not None:
            # Convert t to numpy array
            if isinstance(t, (pd.Series, pd.DataFrame)):
                t = t.values

            # If t is a scalar, create a time vector
            if isinstance(t, (int, float)):
                t = np.arange(self.N, dtype=np.float32) * t

            # Check if the number of samples is the same
            if t.shape[0] != self.N:
                raise ValueError('The number of samples in data and t must be the same.')
            # Check if t is 1d
            if len(t.shape) > 1:
                raise ValueError('t must be 1D.')
            # Check if t is a time vector
            if not np.all(np.diff(t) > 0):
                raise ValueError('t must be a time vector.')

            self.t = t

        # If t is not given, create a time vector
        else:
            self.t = np.arange(self.N, dtype=np.float32)

    def set_initial_state(self, x0: NDArray[np.float32], P0: NDArray[np.float32]) -> None:
        '''
        Sets the initial state of the filter.

        Parameters
        ----------
        x0 : numpy array
            Initial state vector.

        P0 : numpy array
            Initial state covariance matrix.
        '''
        self.x[0] = x0
        self.P[0] = P0

    def predict(self,
                x0: NDArray[np.float32],
                P0: NDArray[np.float32],
                data: Union[NDArray[np.float32], pd.Series, pd.DataFrame],
                inputs: Union[NDArray[np.float32], pd.Series, pd.DataFrame, None] = None,
                t: Union[NDArray[np.float32], pd.Series, pd.DataFrame, float, int, None] = None,
                ) -> None:
        '''
        Predicts the state of the system.

        Parameters
        ----------
        x0 : numpy array
            Initial state vector.

        P0 : numpy array
            Initial state covariance matrix.

        data : numpy array, pandas Series or pandas DataFrame
            Data to be filtered. If data is 2D, the first dimension must be the number of samples
            and the second dimension must be the number of observation.

        inputs : numpy array, pandas Series or pandas DataFrame, optional
            Inputs. If inputs is 2D, the first dimension must be the number of samples and the
            second dimension must be the number of inputs. The default is None.

        t : numpy array, pandas Series, pandas DataFrame, int, float, optional
            Time vector. If t is a scalar, it will be considered the time step and a time vector
            will be created. The default is None.
        '''
        self._mountdata(data, inputs, t)
        self._allocate_arrays()
        self.set_initial_state(x0, P0)
        self._predict()



    def _allocate_arrays(self) -> None:
        '''
        Allocates arrays for the filter.
        '''
        # State Covariance Matrix
        self.P: NDArray[np.float32] = np.zeros((self.N, self.K, self.K), dtype=np.float32)

        # Prediction Error Covariance Matrix
        self.R: NDArray[np.float32] = np.zeros((self.N, self.M, self.M), dtype=np.float32)

        # Kalman Gain
        self.K: NDArray[np.float32] = np.zeros((self.N, self.K, self.M), dtype=np.float32)

        # State Vector
        self.x: NDArray[np.float32] = np.zeros((self.N, self.K), dtype=np.float32)

        # Predicted State Vector
        self.xp: NDArray[np.float32] = np.zeros((self.N, self.K), dtype=np.float32)

        # Output prediction
        self.yp: NDArray[np.float32] = np.zeros((self.N, self.M), dtype=np.float32)

        # Innovation
        self.e: NDArray[np.float32] = np.zeros((self.N, self.M), dtype=np.float32)

    def _predict(self) -> None:
        '''
        Predicts the state of the system.
        '''

        for i in range(1, self.N):
            # Predict the observations
            self.yp[i], self.R[i], self.K[i] = self._output_prediction(self.xp[i-1], self.P[i-1], i)

            # Update the state
            self.x[i], self.P[i] = self._data_update(self.xp[i], self.P[i], i)

            # Time update
            self.xp[i], self.P[i] = self._time_update(self.x[i-1], self.P[i-1], i)

            # Innovation equation
            self.e[i] = self.data[i] - self.yp[i]



    def _time_update(self,
                     x: NDArray[np.float32],
                     P: NDArray[np.float32],
                     i: int,
                     ) -> Union[NDArray[np.float32],
                                NDArray[np.float32]]:
        '''
        Time update step.

        Parameters
        ----------
        x : numpy array
            State vector.

        P : numpy array
            State covariance matrix.

        i : int
            Current time step.

        Returns
        -------
        x : numpy array
            Updated state vector.

        P : numpy array
            Updated state covariance matrix.
        '''
        raise NotImplementedError('This method must be implemented in the derived class.')

    def _data_update(self,
                        x: NDArray[np.float32],
                        P: NDArray[np.float32],
                        i: int,
                        ) -> Union[NDArray[np.float32],
                                    NDArray[np.float32]]:
            '''
            Data update step.

            Parameters
            ----------
            x : numpy array
                State vector.

            P : numpy array
                State covariance matrix.

            i : int
                Current time step.

            Returns
            -------
            x : numpy array
                Updated state vector.

            P : numpy array
                Updated state covariance matrix.
            '''
            raise NotImplementedError('This method must be implemented in the derived class.')

    def _output_prediction(self,
                    x: NDArray[np.float32],
                    P: NDArray[np.float32],
                    i: int,
                    ) -> Union[NDArray[np.float32],
                                NDArray[np.float32],
                                NDArray[np.float32]]:
        '''
        Output prediction step.

        Parameters
        ----------
        x : numpy array
            State vector.

        P : numpy array
            State covariance matrix.

        i : int
            Current time step.

        Returns
        -------
        y:  numpy array
            Innovation.

        R:  numpy array
            Prediction error covariance matrix.

        K:  numpy array
            Kalman Gain.

        '''

        raise NotImplementedError('This method must be implemented in the derived class.')

    def nllikelihood(self) -> NDArray[np.float32]:
        '''
        Computes the loglikelihood of the filter.

        Returns
        -------
        loglikelihood : numpy array
            Loglikelihood of the filter.
        '''
        
        nll = 0
        for i in range(self.N):
            nll += np.log(np.linalg.det(self.R[i])) + self.e[i] @ np.linalg.inv(self.R[i]) @ self.e[i]

        nll = 0.5 * self.M * self.N * np.log(2 * np.pi) + 0.5 * nll
        return nll

