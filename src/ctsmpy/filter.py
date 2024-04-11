from model import Model
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad_vec
from scipy.linalg import expm
from statsmodels.graphics.tsaplots import plot_acf
from tqdm import tqdm

class Forecaster():
    def __init__(self, parameter_filename: str, data_filename: str) -> tuple[float, np.ndarray, np.ndarray]:

        self.number_of_subsamples = 10
        self.k = 0
        self.model = Model(parameter_filename, data_filename)
        self.N = self.model.get_number_of_obserrvations()
        self.T = np.zeros(self.N)
        self.Tinner = np.zeros(self.N)
        T0, Tinner0 = self.model.get_initial_state()
        self.T[0] = T0
        self.Tinner[0] = Tinner0

        u, _ = self.model.get_u_alpha()
        A = self.model.A(0, u, np.zeros(u.shape[0]), [0, 1])
        sigma = self.model.sigma(0, u, np.zeros(u.shape[0]), [0, 1])
        self.P = quad_vec(lambda s: expm(A*s) @ sigma @ sigma.T @ expm(A*s).T, 0, 1)[0]

        obs = self.step(np.array([0, 0, 0]))

        return obs

    def step(self, action: ArrayLike) -> tuple[float, np.ndarray, np.ndarray]:
        t_span = [self.k, self.k + 1]

        u, alpha = self.model.get_u_alpha(action)

        T = self.T[self.k]
        Tinner = self.Tinner[self.k]

        for i in range(self.number_of_subsamples):
            A = self.model.A(self.k + i/self.number_of_subsamples, u, alpha, t_span)
            Phi = expm(A*1/self.number_of_subsamples)
            f = self.model.f(self.k + i/self.number_of_subsamples, [T, Tinner], u, alpha, t_span)
            B = self.model.B(self.k + i/self.number_of_subsamples, u, alpha, t_span)
            [T, Tinner] = [T, Tinner] - np.linalg.inv(A) @ B @ alpha*1/self.number_of_subsamples + np.linalg.inv(A) @ (Phi-np.eye(2)) @ (np.linalg.inv(A) @ B @ alpha + f)

        solP = solve_ivp(self.dP, t_span, self.P.flatten(), args=(u, alpha, t_span), method='LSODA', atol=1e-12, rtol=1e-12)
        self.k += 1
        self.T[self.k] = T
        self.Tinner[self.k] = Tinner
        self.P = solP['y'][:,-1].reshape((2,2))

        weather, demand = self.model.get_observation()

        return (self.T[self.k], weather, demand)

    def dP(self, t: float, P: np.ndarray, u: ArrayLike, alpha: ArrayLike, t_span: ArrayLike) -> np.ndarray:
        A = self.model.A(t, u, alpha, t_span)
        sigma = self.model.sigma(t, u, alpha, t_span)
        P = P.reshape((2,2))
        return (A@P + P@A.T + sigma@sigma.T).flatten()


class Filter(Forecaster):
    def __init__(self, parameter_filename: str, data_filename: str):
        super().__init__(parameter_filename, data_filename)
        self.Tpred = np.zeros(self.N)
        self.Tinnerpred = np.zeros(self.N)
        self.Tpred[0] = self.T[0]
        self.Tinnerpred[0] = self.Tinner[0]
        self.Tpred[1] = self.T[1]
        self.Tinnerpred[1] = self.Tinner[1]
        self.Y = self.model.get_Y()
        self.data_update()

    def state_prediction(self):
        _ = self.step([0, 0, 0])
        self.Tpred[self.k] = self.T[self.k]
        self.Tinnerpred[self.k] = self.Tinner[self.k]

    def data_update(self):
        R = self.model.C @ self.P @ self.model.C.T + self.model.S
        eps = self.Y[self.k] - self.Tpred[self.k]
        K = self.P @ self.model.C.T/R
        update = np.array([[self.Tpred[self.k]], [self.Tinnerpred[self.k]]]) + K*eps
        self.T[self.k] = update[0,0]
        self.Tinner[self.k] = update[1,0]
        self.P = self.P - K @ (R*K.T)

    def run_filter(self, simulate=False):
        for i in tqdm(range(1, self.N - 1)):
            self.state_prediction()
            if not simulate:
                self.data_update()
        self.plot()

    def plot(self):
        res = self.Y - self.Tpred
        print(np.mean(res**2))
        plt.figure()
        plt.plot(res, '*', label='Residuals')
        plt.legend()
        plot_acf(res)
        plt.show()