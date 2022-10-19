import torch
import os
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

class Parameters:
    beta = 0.01
    gamma = 0.05
    N = 100.0


class TrainArgs:
    iteration = 2000
    epoch_step = 100
    test_step = 500
    initial_lr = 0.001
    main_path = "."


class Config:
    def __init__(self):
        self.model_name = "SIR_Fourier"
        self.curve_names = ["S", "I", "R"]
        self.params = Parameters
        self.args = TrainArgs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 0

        self.T = 100
        self.T_unit = 1e-2
        self.T_N = int(self.T / self.T_unit)

        self.prob_dim = 3
        self.y0 = np.asarray([50.0, 49.0, 1.0])
        self.t = np.asarray([i * self.T_unit for i in range(self.T_N)])
        self.t_torch = torch.tensor(self.t, dtype=torch.float32).to(self.device)
        self.x = torch.tensor(np.asarray([[[i * self.T_unit] * self.prob_dim for i in range(self.T_N)]]),
                              dtype=torch.float32).to(self.device)
        # print(self.x.shape)
        self.truth = odeint(self.pend, self.y0, self.t)

        self.modes = 64  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.width = 16
        self.fc_map_dim = 128

    def pend(self, y, t):
        dydt = np.asarray([
            - self.params.beta * y[0] * y[1],
            self.params.beta * y[0] * y[1] - self.params.gamma * y[1],
            self.params.gamma * y[1]
        ])
        return dydt
