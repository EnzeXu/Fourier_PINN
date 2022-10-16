import numpy as np
import time
import torch
import random
import os
import torch.nn.functional as F
from torch import nn


from utils import draw_two_dimension, draw_three_dimension, MultiSubplotDraw
from config_Turing import Config


class SpectralConv3d(nn.Module):
    def __init__(self, config):
        super(SpectralConv3d, self).__init__()

        self.config = config
        self.in_channels = self.config.width
        self.out_channels = self.config.width
        self.modes1 = self.config.modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = self.config.modes2
        self.modes3 = self.config.modes3

        self.scale = (1 / (self.in_channels * self.out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels, self.config.modes1, self.config.modes2,
                                    self.config.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels, self.config.modes1, self.config.modes2,
                                    self.config.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels, self.config.modes1, self.config.modes2,
                                    self.config.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels, self.config.modes1, self.config.modes2,
                                    self.config.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.config.modes1, :self.config.modes2, :self.config.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.config.modes1, :self.config.modes2, :self.config.modes3], self.weights1)
        out_ft[:, :, -self.config.modes1:, :self.config.modes2, :self.config.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.config.modes1:, :self.config.modes2, :self.config.modes3], self.weights2)
        out_ft[:, :, :self.config.modes1, -self.config.modes2:, :self.config.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.config.modes1, -self.config.modes2:, :self.config.modes3], self.weights3)
        out_ft[:, :, -self.config.modes1:, -self.config.modes2:, :self.config.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.config.modes1:, -self.config.modes2:, :self.config.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNO3d(nn.Module):
    def __init__(self, config):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """
        self.config = config
        self.modes1 = self.config.modes1
        self.modes2 = self.config.modes2
        self.modes3 = self.config.modes3
        self.width = self.config.width
        self.padding = 6  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.config)
        self.conv1 = SpectralConv3d(self.config)
        self.conv2 = SpectralConv3d(self.config)
        self.conv3 = SpectralConv3d(self.config)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # grid = self.get_grid(x.shape, x.device)
        # x = grid

        # x = torch.cat((x, grid), dim=-1)
        # print("cp1", x.shape)
        x = self.fc0(x)
        # print("cp2", x.shape)
        x = x.permute(0, 4, 1, 2, 3)
        # print("cp3", x.shape)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        # print("cp4", x1.shape)
        x2 = self.w0(x)
        # print("cp5", x2.shape)
        x = x1 + x2
        x = F.gelu(x)
        # print("cp6", x.shape)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        # print("cp7", x.shape)
        # x = x[..., :-self.padding]
        # print("cp8", x.shape)
        x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic
        # print("cp9", x.shape)
        x = self.fc1(x)
        # print("cp10", x.shape)
        x = F.gelu(x)
        # print("cp11", x.shape)
        x = self.fc2(x)
        # print("cp12", x.shape)
        return x

    @staticmethod
    def get_grid(shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


class FourierModel(nn.Module):
    def __init__(self, config):
        super(FourierModel, self).__init__()
        self.time_string = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
        self.config = config
        self.setup_seed(self.config.seed)

        # self.f_model_u = FNO3d(config)
        # self.f_model_v = FNO3d(config)
        self.f_model = FNO3d(config)

        # self.fc0 = nn.Linear(self.config.prob_dim, self.config.width)  # input channel is 2: (a(x), x)

        # self.conv0 = SpectralConv1d(self.config)
        # self.conv1 = SpectralConv1d(self.config)
        # self.conv2 = SpectralConv1d(self.config)
        # self.conv3 = SpectralConv1d(self.config)
        # self.w0 = nn.Conv1d(self.config.width, self.config.width, 1)
        # self.w1 = nn.Conv1d(self.config.width, self.config.width, 1)
        # self.w2 = nn.Conv1d(self.config.width, self.config.width, 1)
        # self.w3 = nn.Conv1d(self.config.width, self.config.width, 1)

        # self.fc1 = nn.Linear(self.config.width, self.config.fc_map_dim)
        # self.fc2 = nn.Linear(self.config.fc_map_dim, self.config.prob_dim)

        self.criterion = torch.nn.MSELoss().to(
            self.config.device)  # self.criterion = torch.nn.MSELoss("sum").to(self.config.device)

        self.y_tmp = None
        self.epoch_tmp = None
        self.loss_record_tmp = None

        self.figure_save_path_folder = "{0}/figure/{1}_{2}/".format(self.config.args.main_path, self.config.model_name,
                                                                    self.time_string)
        if not os.path.exists(self.figure_save_path_folder):
            os.makedirs(self.figure_save_path_folder)
        self.default_colors = ["red", "blue", "green", "orange", "cyan", "purple", "pink", "indigo", "brown", "grey"]

        print("using {}".format(str(self.config.device)))
        print("iteration = {}".format(self.config.args.iteration))
        print("epoch_step = {}".format(self.config.args.epoch_step))
        print("test_step = {}".format(self.config.args.test_step))
        print("model_name = {}".format(self.config.model_name))
        print("time_string = {}".format(self.time_string))
        self.truth_loss()

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def ode_gradient(self, y):
        # y: 1 * T_N * N * M * 2
        y = y[0]
        shapes = y.shape
        reaction_part = torch.zeros([shapes[0], shapes[1], shapes[2], 2]).to(self.config.device)
        reaction_part[:, :, :, 0] = self.config.params.c1 - self.config.params.c_1 * y[:, :, :,
                                                                                     0] + self.config.params.c3 * (
                                                y[:, :, :, 0] ** 2) * y[:, :, :, 1]
        reaction_part[:, :, :, 1] = self.config.params.c2 - self.config.params.c3 * (y[:, :, :, 0] ** 2) * y[:, :, :, 1]

        y_from_left = torch.roll(y, 1, 2)
        y_from_left[:, :, :1] = y[:, :, :1]
        y_from_right = torch.roll(y, -1, 2)
        y_from_right[:, :, -1:] = y[:, :, -1:]

        y_from_top = torch.roll(y, 1, 1)
        y_from_top[:, :1, :] = y[:, :1, :]
        y_from_bottom = torch.roll(y, -1, 1)
        y_from_bottom[:, -1:, :] = y[:, -1:, :]

        diffusion_part = torch.zeros([shapes[0], shapes[1], shapes[2], 2]).to(self.config.device)
        diffusion_part[:, :, :, 0] = self.config.params.d1 * (((y_from_left[:, :, :, 0] + y_from_right[:, :, :, 0] - y[
                                                                                                                     :,
                                                                                                                     :,
                                                                                                                     :,
                                                                                                                     0] * 2) / (
                                                                           self.config.params.l ** 2)) + ((y_from_top[:,
                                                                                                           :, :,
                                                                                                           0] + y_from_bottom[
                                                                                                                :, :, :,
                                                                                                                0] - y[
                                                                                                                     :,
                                                                                                                     :,
                                                                                                                     :,
                                                                                                                     0] * 2) / (
                                                                                                                      self.config.params.w ** 2)))
        diffusion_part[:, :, :, 1] = self.config.params.d2 * (((y_from_left[:, :, :, 1] + y_from_right[:, :, :, 1] - y[
                                                                                                                     :,
                                                                                                                     :,
                                                                                                                     :,
                                                                                                                     1] * 2) / (
                                                                           self.config.params.l ** 2)) + ((y_from_top[:,
                                                                                                           :, :,
                                                                                                           1] + y_from_bottom[
                                                                                                                :, :, :,
                                                                                                                1] - y[
                                                                                                                     :,
                                                                                                                     :,
                                                                                                                     :,
                                                                                                                     1] * 2) / (
                                                                                                                      self.config.params.w ** 2)))

        y_t_theory = reaction_part + diffusion_part

        y_t = torch.gradient(y, spacing=(self.config.t_torch,), dim=0)[0]

        return y_t - y_t_theory

    def loss(self, y):
        y0_pred = y[0, 0]
        y0_true = self.config.y0

        ode_y = self.ode_gradient(y)
        zeros_nD = torch.zeros([self.config.T_N, self.config.params.N, self.config.params.M, self.config.prob_dim]).to(
            self.config.device)

        loss1 = 1 * self.criterion(y0_pred, y0_true)
        loss2 = 1e-1 * self.criterion(ode_y, zeros_nD)

        loss3 = 1 * (self.criterion(torch.abs(y - 0.1), y - 0.1) + self.criterion(torch.abs(6.5 - y), 6.5 - y))
        # loss4 = self.criterion(1e-3 / (y[0, :, :] ** 2 + 1e-10), zeros_nD)
        # self.criterion(1e-3 / (ode_1 ** 2 + 1e-10), zeros_1D) + self.criterion(1e-3 / (ode_2 ** 2 + 1e-10), zeros_1D) + self.criterion(1e-3 / (ode_3 ** 2 + 1e-10), zeros_1D)
        # loss5 = self.criterion(torch.abs(u_0 - v_0), u_0 - v_0)

        loss = loss1 + loss2 + loss3
        loss_list = [loss1, loss2, loss3]
        return loss, loss_list

    def truth_loss(self):
        y_truth = self.config.truth.reshape(
            [1, self.config.T_N, self.config.params.N, self.config.params.M, self.config.prob_dim])
        # print("y_truth max:", torch.max(y_truth))
        # print("y_truth min:", torch.min(y_truth))
        tl, tl_list = self.loss(y_truth)
        loss_print_part = " ".join(
            ["Loss_{0:d}:{1:.8f}".format(i + 1, loss_part.item()) for i, loss_part in enumerate(tl_list)])
        print("Ground truth has loss: Loss:{0:.8f} {1}".format(tl.item(), loss_print_part))

    def train_model(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.args.initial_lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1 / (e / 1000 + 1))
        self.train()

        start_time = time.time()
        start_time_0 = start_time
        loss_record = []

        for epoch in range(1, self.config.args.iteration + 1):
            optimizer.zero_grad()

            # u = self.f_model_u(self.config.x)
            # v = self.f_model_v(self.config.x)
            y = self.f_model(self.config.x)
            # print("y shape:", y.shape)
            # print("v shape:", v.shape)
            # y = torch.concat([u, v], dim=-1)
            loss, loss_list = self.loss(y)
            loss_record.append(loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % self.config.args.epoch_step == 0:
                now_time = time.time()
                loss_print_part = " ".join(
                    ["Loss_{0:d}:{1:.8f}".format(i + 1, loss_part.item()) for i, loss_part in enumerate(loss_list)])
                print(
                    "Epoch [{0:05d}/{1:05d}] Loss:{2:.8f} {3} Lr:{4:.8f} Time:{5:.6f}s ({6:.2f}min in total, {7:.2f}min remains)".format(
                        epoch, self.config.args.iteration, loss.item(), loss_print_part,
                        optimizer.param_groups[0]["lr"], now_time - start_time, (now_time - start_time_0) / 60.0,
                        (now_time - start_time_0) / 60.0 / epoch * (self.config.args.iteration - epoch)))
                start_time = now_time

                if epoch % self.config.args.test_step == 0:
                    self.y_tmp = y
                    self.epoch_tmp = epoch
                    self.loss_record_tmp = loss_record
                    self.test_model()

    def test_model(self):
        u_draw_all = self.y_tmp[0, :, :, :, 0].reshape(self.config.T_N,
                                                       self.config.params.N * self.config.params.M).cpu().detach().numpy().swapaxes(
            0, 1)[[10 * i for i in range(10)]]
        u_draw_all_truth = self.config.truth[:, :, :, 0].reshape(self.config.T_N,
                                                                 self.config.params.N * self.config.params.M).cpu().detach().numpy().swapaxes(
            0, 1)[[10 * i for i in range(10)]]
        v_draw_all = self.y_tmp[0, :, :, :, 1].reshape(self.config.T_N,
                                                       self.config.params.N * self.config.params.M).cpu().detach().numpy().swapaxes(
            0, 1)[[10 * i for i in range(10)]]
        v_draw_all_truth = self.config.truth[:, :, :, 1].reshape(self.config.T_N,
                                                                 self.config.params.N * self.config.params.M).cpu().detach().numpy().swapaxes(
            0, 1)[[10 * i for i in range(10)]]
        x_draw = self.config.t
        draw_n = len(u_draw_all)
        save_path_2D = "{}/{}_{}_epoch={}_2D.png".format(self.figure_save_path_folder, self.config.model_name,
                                                         self.time_string, self.epoch_tmp)

        m = MultiSubplotDraw(row=1, col=2, fig_size=(16, 6), tight_layout_flag=True, show_flag=True, save_flag=True,
                             save_path=save_path_2D)
        m.add_subplot(
            y_lists=np.concatenate([u_draw_all, u_draw_all_truth], axis=0),
            x_list=x_draw,
            color_list=[self.default_colors[0]] * draw_n + [self.default_colors[1]] * draw_n,
            line_style_list=["solid"] * draw_n + ["dashed"] * draw_n,
            fig_title="{}_{}_U_epoch={}_2D".format(self.config.model_name, self.time_string, self.epoch_tmp),
            line_width=0.5)
        m.add_subplot(
            y_lists=np.concatenate([v_draw_all, v_draw_all_truth], axis=0),
            x_list=x_draw,
            color_list=[self.default_colors[0]] * draw_n + [self.default_colors[1]] * draw_n,
            line_style_list=["solid"] * draw_n + ["dashed"] * draw_n,
            fig_title="{}_{}_V_epoch={}_2D".format(self.config.model_name, self.time_string, self.epoch_tmp),
            line_width=0.5, )
        m.draw()

        # draw_two_dimension(
        #     y_lists=np.concatenate([u_draw_all, u_draw_all_truth], axis=0),
        #     x_list=x_draw,
        #     color_list=[self.default_colors[0]] * draw_n + [self.default_colors[1]] * draw_n,
        #     line_style_list=["solid"] * draw_n + ["dashed"] * draw_n,
        #     fig_title="{}_{}_U_epoch={}_2D".format(self.config.model_name, self.time_string, self.epoch_tmp),
        #     fig_size=(8, 6),
        #     line_width=0.5,
        #     show_flag=True,
        #     save_flag=True,
        #     save_path=save_path_2D_u,
        # )
        # save_path_2D_v = "{}/{}_v_{}_epoch={}_2D.png".format(self.figure_save_path_folder, self.config.model_name, self.time_string, self.epoch_tmp)
        # draw_two_dimension(
        #     y_lists=np.concatenate([v_draw_all, v_draw_all_truth], axis=0),
        #     x_list=x_draw,
        #     color_list=[self.default_colors[0]] * draw_n + [self.default_colors[1]] * draw_n,
        #     line_style_list=["solid"] * draw_n + ["dashed"] * draw_n,
        #     fig_title="{}_{}_V_epoch={}_2D".format(self.config.model_name, self.time_string, self.epoch_tmp),
        #     fig_size=(8, 6),
        #     line_width=0.5,
        #     show_flag=True,
        #     save_flag=True,
        #     save_path=save_path_2D_v,
        # )
        print("2D Figure is saved to {}".format(save_path_2D))
        # y_draw = self.y_tmp[0].cpu().detach().numpy().swapaxes(0, 1)
        # x_draw = self.config.t
        # y_draw_truth = self.config.truth.swapaxes(0, 1)
        # save_path_2D = "{}/{}_{}_epoch={}_2D.png".format(self.figure_save_path_folder, self.config.model_name, self.time_string, self.epoch_tmp)
        # save_path_3D = "{}/{}_{}_epoch={}_3D.png".format(self.figure_save_path_folder, self.config.model_name, self.time_string, self.epoch_tmp)
        # draw_two_dimension(
        #     y_lists=np.concatenate([y_draw, y_draw_truth], axis=0),
        #     x_list=x_draw,
        #     color_list=self.default_colors[: 2 * self.config.prob_dim],
        #     legend_list=self.config.curve_names + ["{}_true".format(item) for item in self.config.curve_names],
        #     line_style_list=["solid"] * self.config.prob_dim + ["dashed"] * self.config.prob_dim,
        #     fig_title="{}_{}_epoch={}_2D".format(self.config.model_name, self.time_string, self.epoch_tmp),
        #     fig_size=(8, 6),
        #     show_flag=True,
        #     save_flag=True,
        #     save_path=save_path_2D,
        # )
        # print("2D Figure is saved to {}".format(save_path_2D))

        # draw_three_dimension(
        #     lists=[y_draw, y_draw_truth],
        #     legend_list=["pred", "true"],
        #     color_list=self.default_colors[:2],
        #     line_style_list=["solid", "dashed"],
        #     fig_title="{}_{}_epoch={}_3D".format(self.config.model_name, self.time_string, self.epoch_tmp),
        #     alpha=0.7,
        #     show_flag=True,
        #     save_flag=True,
        #     save_path=save_path_3D,
        #     fig_size=(8, 6),
        #     line_width=1.0,
        #     lim_adaptive_flag=True
        # )
        # print("3D Figure is saved to {}".format(save_path_3D))

        # y_draw = self.y_tmp[0, -1]
        # y_draw_truth = self.config.truth[-1]
        # print("Pred: {}".format(y_draw.shape))
        # self.config.draw_turing(y_draw)
        # print("True: {}".format(y_draw_truth.shape))
        # self.config.draw_turing(y_draw_truth)
        u = self.y_tmp[0, :, :, :, 0].cpu().detach().numpy()
        v = self.y_tmp[0, :, :, :, 1].cpu().detach().numpy()
        u_last = u[-1]
        v_last = v[-1]
        u_true = self.config.truth[:, :, :, 0].cpu().detach().numpy()
        v_true = self.config.truth[:, :, :, 1].cpu().detach().numpy()
        u_last_true = u_true[-1]
        v_last_true = v_true[-1]
        save_path_map_all = "{}/{}_{}_epoch={}_map_all.png".format(self.figure_save_path_folder, self.config.model_name,
                                                                   self.time_string, self.epoch_tmp)
        save_path_map_pred_only = "{}/{}_{}_epoch={}_map_pred_only.png".format(self.figure_save_path_folder,
                                                                               self.config.model_name, self.time_string,
                                                                               self.epoch_tmp)
        m = MultiSubplotDraw(row=2, col=2, fig_size=(16, 14), tight_layout_flag=True, save_flag=True,
                             save_path=save_path_map_all)
        m.add_subplot_turing(
            matrix=u_last,
            v_max=u_last_true.max(),
            v_min=u_last_true.min(),
            fig_title_size=10,
            number_label_size=10,
            fig_title="{}_{}_U_pred_epoch={}".format(self.config.model_name, self.time_string, self.epoch_tmp))
        m.add_subplot_turing(
            matrix=v_last,
            v_max=v_last_true.max(),
            v_min=v_last_true.min(),
            fig_title_size=10,
            number_label_size=10,
            fig_title="{}_{}_V_pred_epoch={}".format(self.config.model_name, self.time_string, self.epoch_tmp))
        m.add_subplot_turing(
            matrix=u_last_true,
            v_max=u_last_true.max(),
            v_min=u_last_true.min(),
            fig_title_size=10,
            number_label_size=10,
            fig_title="{}_{}_U_true".format(self.config.model_name, self.time_string))
        m.add_subplot_turing(
            matrix=v_last_true,
            v_max=v_last_true.max(),
            v_min=v_last_true.min(),
            fig_title_size=10,
            number_label_size=10,
            fig_title="{}_{}_V_true".format(self.config.model_name, self.time_string))
        m.draw()

        m = MultiSubplotDraw(row=1, col=2, fig_size=(16, 7), tight_layout_flag=True, show_flag=False, save_flag=True,
                             save_path=save_path_map_pred_only)
        m.add_subplot_turing(
            matrix=u_last,
            v_max=u_last_true.max(),
            v_min=u_last_true.min(),
            fig_title_size=10,
            number_label_size=10,
            fig_title="{}_{}_U_pred_epoch={}".format(self.config.model_name, self.time_string, self.epoch_tmp))
        m.add_subplot_turing(
            matrix=v_last,
            v_max=v_last_true.max(),
            v_min=v_last_true.min(),
            fig_title_size=10,
            number_label_size=10,
            fig_title="{}_{}_V_pred_epoch={}".format(self.config.model_name, self.time_string, self.epoch_tmp))
        m.draw()

        self.draw_loss_multi(self.loss_record_tmp, [1.0, 0.5, 0.25, 0.125])

    @staticmethod
    def draw_loss_multi(loss_list, last_rate_list):
        m = MultiSubplotDraw(row=1, col=len(last_rate_list), fig_size=(8 * len(last_rate_list), 6),
                             tight_layout_flag=True, show_flag=True, save_flag=False, save_path=None)
        for one_rate in last_rate_list:
            m.add_subplot(
                y_lists=[loss_list[-int(len(loss_list) * one_rate):]],
                x_list=range(len(loss_list) - int(len(loss_list) * one_rate) + 1, len(loss_list) + 1),
                color_list=["blue"],
                line_style_list=["solid"],
                fig_title="Loss - lastest ${}$% - epoch ${}$ to ${}$".format(int(100 * one_rate), len(loss_list) - int(
                    len(loss_list) * one_rate) + 1, len(loss_list)),
                fig_x_label="epoch",
                fig_y_label="loss",
            )
        m.draw()

if __name__ == "__main__":
    config = Config()
    model = FourierModel(config).to(config.device)
    model.train_model()



