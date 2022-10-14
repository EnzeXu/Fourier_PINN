import torch
import numpy as np
from scipy.integrate import odeint


class Parameters:
    alpha = 1.00
    beta = 3.00
    gamma = 0.30
    e = 0.333 * 0.30


class TrainArgs:
    iteration = 1000000
    epoch_step = 1000
    test_step = 10000
    initial_lr = 0.01
    main_path = "."


class Config:
    def __init__(self):
        self.model_name = "PP_Fourier"
        self.curve_names = ["U", "V"]
        self.params = Parameters
        self.args = TrainArgs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 0

        self.T = 12
        self.T_unit = 6e-4
        self.T_N = int(self.T / self.T_unit)

        self.prob_dim = 2
        self.y0 = np.asarray([10.0, 5.0])
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
        dy_dt = np.asarray([
            (self.params.alpha * y[0]) - self.params.gamma * y[0] * y[1],
            - self.params.beta * y[1] + self.params.e * y[0] * y[1]
        ])
        return dy_dt
