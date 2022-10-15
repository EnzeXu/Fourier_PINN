import torch
import numpy as np
from scipy.integrate import odeint


class Parameters:
    rho = 14  # 28.0
    sigma = 10.0
    beta = 2.667


class TrainArgs:
    iteration = 1000000
    epoch_step = 10
    test_step = 100
    initial_lr = 0.005
    main_path = "."


class Config:
    def __init__(self):
        self.model_name = "Lorenz_Fourier"
        self.curve_names = ["X", "Y", "Z"]
        self.params = Parameters
        self.args = TrainArgs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 0

        self.T = 5
        self.T_unit = 1e-4
        self.T_N = int(self.T / self.T_unit)

        self.prob_dim = 3
        self.y0 = np.asarray([6.0, 6.0, 15.0])
        self.t = np.asarray([i * self.T_unit for i in range(self.T_N)])
        self.t_torch = torch.tensor(self.t, dtype=torch.float32).to(self.device)
        self.x = torch.tensor(np.asarray([[[i * self.T_unit] * self.prob_dim for i in range(self.T_N)]]),
                              dtype=torch.float32).to(self.device)
        # print(self.x.shape)
        self.truth = odeint(self.pend, self.y0, self.t)
        print("Truth X: min={0:.6f} max={1:.6f}".format(np.min(self.truth[:, 0]), np.max(self.truth[:, 0])))
        print("Truth Y: min={0:.6f} max={1:.6f}".format(np.min(self.truth[:, 1]), np.max(self.truth[:, 1])))
        print("Truth Z: min={0:.6f} max={1:.6f}".format(np.min(self.truth[:, 2]), np.max(self.truth[:, 2])))

        self.modes = 64  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.width = 16
        self.fc_map_dim = 128

    def pend(self, y, t):
        dydt = np.asarray([
            self.params.sigma * (y[1] - y[0]),
            y[0] * (self.params.rho - y[2]) - y[1],
            y[0] * y[1] - self.params.beta * y[2]
        ])
        return dydt