import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

class Model():
    # Class for SDE model of temperature in a transformer station
    # The model is a two state SDE model with the following states
    # dT = 1/C*(1/Rit*(Tinner - T) + 1/(Rot + (w1*ws1+w2*ws2+w3*ws3))*(Tout - T)
    # + (s1*ts1+s2*ts2+s3*ts3+s4*ts4+s5*ts5)*solar + p*precip)*dt 
    # + (exp(omega1) + d*(s1*ts1+s2*ts2+s3*ts3+s4*ts4+s5*ts5)*solar + d2*Toutnorm)*dw1
    #
    # dTinner ~ 1/Ci*(1/Rit*(T - Tinner) + L*(Phase1^2+Phase2^2+Phase3^2) + c + N1*(p1dot*Phase1+p2dot*Phase2+p3dot*Phase3) + N2*(p1dott*Phase1+p2dott*Phase2+p3dott*Phase3))*dt
    # + (di*Inorm + exp(omega2))*dw2
    def __init__(self, parameter_filename: str, data_filename: str):
        parameters = pd.read_csv(parameter_filename, index_col=0)
        self.par = {}
        for i in range(0, parameters.shape[0]):
            self.par[parameters.index[i]] = parameters.loc[parameters.index[i]].values[0]
        self.C = np.array([[1, 0]])
        self.S  = np.exp(self.par['eta'])

        data = pd.read_csv(data_filename)
        self.weather_inputs = ['ws1', 'ws2', 'ws3', 'ts1', 'ts2', 'ts3', 'ts4', 'ts5', 'Tout', 'solar', 'precip']
        self.N = data.shape[0]
        self.n_features = data.shape[1] - 2
        self.n_weather_features = len(self.weather_inputs)
        self.weather_data = data[self.weather_inputs]
        self.demand = data[['Phase1', 'Phase2', 'Phase3']]
        self.Y = data['yT'].values
        self.load_history = self.demand.iloc[0].values
        self.k = 0

    def get_initial_state(self) -> tuple[float, float]:
        return self.par['T0'], self.par['Tinner0']

    def get_number_of_obserrvations(self) -> int:
        return self.N

    def get_Y(self) -> np.ndarray:
        return self.Y

    def f(self, t: float, X: ArrayLike, u: ArrayLike, alpha: ArrayLike, t_span: ArrayLike) -> np.ndarray:
        tk = (t - t_span[0])/(t_span[1] - t_span[0])
        wind_factor = self.par['w1']*(u[0]+tk*alpha[0]) + self.par['w2']*(u[1]+tk*alpha[1]) + self.par['w3']*(u[2]+tk*alpha[2])
        solar_factor = self.par['s1']*(u[3]+tk*alpha[3]) + self.par['s2']*(u[4]+tk*alpha[4]) + self.par['s3']*(u[5]+tk*alpha[5]) + self.par['s4']*(u[6]+tk*alpha[6]) + self.par['s5']*(u[7]+tk*alpha[7])
        pdot  = (u[14]+tk*alpha[14])*(u[11]+tk*alpha[11]) + (u[15]+tk*alpha[15])*(u[12]+tk*alpha[12]) + (u[16]+tk*alpha[16])*(u[13]+tk*alpha[13])
        pdott = (u[17]+tk*alpha[17])*(u[11]+tk*alpha[11]) + (u[18]+tk*alpha[18])*(u[12]+tk*alpha[12]) + (u[19]+tk*alpha[19])*(u[13]+tk*alpha[13])

        dT = 1/self.par['C']*(1/self.par['Rit']*(X[1] - X[0]) + 1/(self.par['Rot'] + wind_factor)*((u[8]+tk*alpha[8]) - X[0]) + solar_factor*(u[9]+tk*alpha[9]) + self.par['p']*(u[10]+tk*alpha[10]))
        dTinner = 1/self.par['Ci']*(1/self.par['Rit']*(X[0] - X[1]) + self.par['L']*((u[11]+tk*alpha[11])**2 + (u[12]+tk*alpha[12])**2 + (u[13]+tk*alpha[13])**2) + self.par['N1']*pdot + self.par['N2']*pdott + self.par['c'])

        return np.array([dT, dTinner])

    def A(self, t: float, u: ArrayLike, alpha: ArrayLike, t_span: ArrayLike) -> np.ndarray:
        tk = (t - t_span[0])/(t_span[1] - t_span[0])
        wind_factor = self.par['w1']*(u[0]+tk*alpha[0]) + self.par['w2']*(u[1]+tk*alpha[1]) + self.par['w3']*(u[2]+tk*alpha[2])
        return np.array([[-1/(self.par['C']*self.par['Rit']) - 1/(self.par['Rot'] + wind_factor), 1/(self.par['C']*self.par['Rit'])],
                  [1/(self.par['Ci']*self.par['Rit']), -1/(self.par['Ci']*self.par['Rit'])]])

    def B(self, t: float, u: ArrayLike, alpha: ArrayLike, t_span: ArrayLike) -> np.ndarray:
        tk = (t - t_span[0])/(t_span[1] - t_span[0])
        B1 = [-self.par['w1']/(self.par['Rot']+self.par['w1']*(u[0]+tk*alpha[0]))**2, -self.par['w2']/(self.par['Rot']+self.par['w2']*(u[1]+tk*alpha[1]))**2, -self.par['w3']/(self.par['Rot']+self.par['w3']*(u[2]+tk*alpha[2]))**2, self.par['s1']*(u[9]+tk*alpha[9]), self.par['s2']*(u[9]+tk*alpha[9]), self.par['s3']*(u[9]+tk*alpha[9]), self.par['s4']*(u[9]+tk*alpha[9]), self.par['s5']*(u[9]+tk*alpha[9]), 1/(self.par['Rot'] + self.par['w1']*(u[0]+tk*alpha[0]) + self.par['w2']*(u[1]+tk*alpha[1]) + self.par['w3']*(u[2]+tk*alpha[2])), self.par['s1']*(u[3]+tk*alpha[3])+self.par['s2']*(u[4]+tk*alpha[4])+self.par['s3']*(u[5]+tk*alpha[5])+self.par['s4']*(u[6]+tk*alpha[6])+self.par['s5']*(u[7]+tk*alpha[7]), self.par['p'], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        B2 = [0,0,0,0,0,0,0,0,0,0,0,2*self.par['L']*(u[11]+tk*alpha[11]) + self.par['N1']*(u[14]+tk*alpha[14])+self.par['N2']*(u[17]+tk*alpha[17]), 2*self.par['L']*(u[12]+tk*alpha[12])+ self.par['N1']*(u[15]+tk*alpha[15])+self.par['N2']*(u[18]+tk*alpha[18]), 2*self.par['L']*(u[13]+tk*alpha[13])+ self.par['N1']*(u[16]+tk*alpha[16])+self.par['N2']*(u[19]+tk*alpha[19]), self.par['N1']*(u[11]+tk*alpha[11]), self.par['N1']*(u[12]+tk*alpha[12]), self.par['N1']*(u[13]+tk*alpha[13]), self.par['N2']*(u[11]+tk*alpha[11]), self.par['N2']*(u[12]+tk*alpha[12]), self.par['N2']*(u[13]+tk*alpha[13]), 0, 0]
        return np.array([1/self.par['C']*np.array(B1), 1/self.par['Ci']*np.array(B2)])

    def sigma(self, t: float, u: ArrayLike, alpha: ArrayLike, t_span: ArrayLike) -> np.ndarray:
        tk = (t - t_span[0])/(t_span[1] - t_span[0])
        solar_factor = self.par['s1']*(u[3]+tk*alpha[3]) + self.par['s2']*(u[4]+tk*alpha[4]) + self.par['s3']*(u[5]+tk*alpha[5]) + self.par['s4']*(u[6]+tk*alpha[6]) + self.par['s5']*(u[7]+tk*alpha[7])
        sigma1 = np.exp(self.par['omega1']) + self.par['d']*solar_factor*(u[9]+tk*alpha[9]) + self.par['d2']*(u[21]+tk*alpha[21])
        sigma2 = np.exp(self.par['omega2']) + self.par['di']*(u[20]+tk*alpha[20])
        return np.array([[sigma1], [sigma2]])

    def get_u_alpha(self, action: ArrayLike=None):
        u = np.zeros(self.n_features)
        u[:self.n_weather_features] = self.weather_data.iloc[self.k].values
        load = self.demand.iloc[self.k].values + action if action is not None else self.demand.iloc[self.k].values
        u[self.n_weather_features:(self.n_weather_features+3)] = load
        u[(self.n_weather_features+3):(self.n_weather_features+6)] = (load - self.load_history)/0.3717568
        u[(self.n_weather_features+9)] = (np.sum(load)*413.1085 - 215.3195)/(1167.825 - 215.3195)
        u[(self.n_weather_features+10)] = (u[8] + 5.6)/(26.9 + 5.6)

        alpha = np.zeros(self.n_features)
        alpha[:self.n_weather_features] = self.weather_data.iloc[self.k:self.k+2].diff().values[1:]
        alpha[self.n_weather_features:(self.n_weather_features+3)] = self.demand.iloc[self.k+1].values - load
        alpha[(self.n_weather_features+3):(self.n_weather_features+6)] = (self.demand.iloc[self.k+1].values - load)/0.3717568 - u[(self.n_weather_features+3):(self.n_weather_features+6)]
        alpha[(self.n_weather_features+9)] = ((np.sum(load)-np.sum(self.load_history))*413.1085)/(1167.825 - 215.3195)
        alpha[(self.n_weather_features+10)] = alpha[8]/(26.9 + 5.6)

        if action is not None:
            self.load_history = load
            self.k += 1

        return u, alpha

    def get_observation(self):
        return self.weather_data.iloc[self.k].values, self.demand.iloc[self.k]