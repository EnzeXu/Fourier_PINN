import torch
import os
import torchdiffeq
import numpy as np
import matplotlib.pyplot as plt


class Parameters:
    N = 50
    M = 50
    d1 = 1
    d2 = 40
    c1 = 0.1
    c2 = 0.9
    c_1 = 1
    c3 = 1
    l = 1
    w = 1


class TrainArgs:
    iteration = 1000000
    epoch_step = 1
    test_step = 10
    initial_lr = 0.001
    ignore_save_flag = True
    main_path = "."


class Config:
    def __init__(self):
        self.model_name = "Turing_Fourier"
        self.curve_names = ["U", "V"]
        self.params = Parameters
        self.args = TrainArgs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 0

        self.T_before = 30
        self.noise_rate = 0.05
        self.T = 2
        self.T_unit = 2e-3
        self.T_N_before = int(self.T_before / self.T_unit)
        self.T_N = int(self.T / self.T_unit)

        self.prob_dim = 2

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.y0_before = torch.rand([self.params.N, self.params.M, self.prob_dim]).to(self.device) + 2.0
        self.t_before = np.asarray([i * self.T_unit for i in range(self.T_N_before)])
        self.t = np.asarray([i * self.T_unit for i in range(self.T_N)])
        self.t_torch = torch.tensor(self.t, dtype=torch.float32).to(self.device)
        # self.x = torch.tensor(np.asarray([[[i * self.T_unit] * self.prob_dim for i in range(self.T_N)]]), dtype=torch.float32).to(self.device)
        # self.x = torch.tensor(np.asarray([[[i * self.T_unit] * self.prob_dim for i in range(self.T_N)]]), dtype=torch.float32).to(self.device)

        # x_1 = torch.zeros(self.params.N, self.params.M, 2, self.T_N).to(self.device)

        # x_0 = torch.tensor([1.0 / (self.params.N * self.params.M) * i for i in range(self.params.N * self.params.M)]).reshape(self.params.N, self.params.M).to(self.device)
        # x_1 = x_0.repeat(1, self.T_N, 1, 1)
        # x_1 = x_1.permute(2, 3, 0, 1)
        # x_2 = torch.tensor([i * self.T_unit for i in range(self.T_N)]).to(self.device)
        # x = x_1 + x_2
        # x = x.permute(3, 0, 1, 2)
        # self.x = x.reshape([1, self.T_N, self.params.N, self.params.M, 1])
        x = torch.zeros([1, self.T_N, self.params.N, self.params.M, 1]).to(self.device)
        self.x = get_grid(x.shape, x.device)
        print("x shape:", self.x.shape)
        # self.truth = odeint(self.pend, self.y0, self.t)

        truth_path = self.args.main_path + "/saves/turing_truth.npy"
        if os.path.exists(truth_path) and not self.args.ignore_save_flag:
            self.truth = torch.tensor(np.load(truth_path), dtype=torch.float32).to(self.device)
            self.y0 = self.truth[0]
            print("Truth exists. Loading...")
        else:
            truth_before = torchdiffeq.odeint(self.pend, self.y0_before.cpu(), torch.tensor(self.t_before),
                                              method='euler').to(self.device)
            noise = (torch.rand([self.params.N, self.params.M, self.prob_dim]).to(self.device) - 0.5) * self.noise_rate
            self.y0 = torch.abs(truth_before[-1] * (1.0 + noise) + 0.2)
            self.truth = torchdiffeq.odeint(self.pend, self.y0.cpu(), torch.tensor(self.t), method='euler').to(
                self.device)
            np.save(truth_path, self.truth.cpu().detach().numpy())
        # print("y0:")
        # self.draw_turing(self.y0)
        # print("Truth:")
        print("Truth U: max={0:.6f} min={1:.6f}".format(torch.max(self.truth[:, :, :, 0]).item(),
                                                        torch.min(self.truth[:, :, :, 0]).item()))
        print("Truth V: max={0:.6f} min={1:.6f}".format(torch.max(self.truth[:, :, :, 1]).item(),
                                                        torch.min(self.truth[:, :, :, 1]).item()))
        # self.draw_turing(self.truth[-1])

        # self.modes = 64  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        # self.width = 16
        self.modes1 = 12  # 8
        self.modes2 = 12
        self.modes3 = 12
        self.width = 32  # 20
        # self.fc_map_dim = 128

    def pend(self, t, y):
        shapes = y.shape
        reaction_part = torch.zeros([shapes[0], shapes[1], 2])
        reaction_part[:, :, 0] = self.params.c1 - self.params.c_1 * y[:, :, 0] + self.params.c3 * (y[:, :, 0] ** 2) * y[
                                                                                                                      :,
                                                                                                                      :,
                                                                                                                      1]
        reaction_part[:, :, 1] = self.params.c2 - self.params.c3 * (y[:, :, 0] ** 2) * y[:, :, 1]

        y_from_left = torch.roll(y, 1, 1)
        y_from_left[:, :1] = y[:, :1]
        y_from_right = torch.roll(y, -1, 1)
        y_from_right[:, -1:] = y[:, -1:]

        y_from_top = torch.roll(y, 1, 0)
        y_from_top[:1, :] = y[:1, :]
        y_from_bottom = torch.roll(y, -1, 0)
        y_from_bottom[-1:, :] = y[-1:, :]

        diffusion_part = torch.zeros([shapes[0], shapes[1], 2])
        diffusion_part[:, :, 0] = self.params.d1 * (
                    ((y_from_left[:, :, 0] + y_from_right[:, :, 0] - y[:, :, 0] * 2) / (self.params.l ** 2)) + (
                        (y_from_top[:, :, 0] + y_from_bottom[:, :, 0] - y[:, :, 0] * 2) / (self.params.w ** 2)))
        diffusion_part[:, :, 1] = self.params.d2 * (
                    ((y_from_left[:, :, 1] + y_from_right[:, :, 1] - y[:, :, 1] * 2) / (self.params.l ** 2)) + (
                        (y_from_top[:, :, 1] + y_from_bottom[:, :, 1] - y[:, :, 1] * 2) / (self.params.w ** 2)))
        return reaction_part + diffusion_part

    @staticmethod
    def draw_turing(map):
        # map: N * M * 2
        u = map[:, :, 0].cpu().detach().numpy()
        v = map[:, :, 1].cpu().detach().numpy()
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(121)
        im1 = ax1.imshow(u, cmap=plt.cm.jet, aspect='auto')
        ax1.set_title("u")
        cb1 = plt.colorbar(im1, shrink=1)

        ax2 = fig.add_subplot(122)
        im2 = ax2.imshow(v, cmap=plt.cm.jet, aspect='auto')
        ax2.set_title("v")
        cb2 = plt.colorbar(im2, shrink=1)
        plt.tight_layout()
        plt.show()


def get_grid(shape, device):
    batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
    gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
    gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
    return torch.cat((gridx, gridy, gridz), dim=-1).to(device)