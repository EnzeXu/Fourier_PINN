import matplotlib.pyplot as plt
import numpy as np
import time


def add_time(func):
    def wrapper(*args, **kwargs):
        if func.__name__ == "myprint":
            with open(args[1], "a") as f:
                f.write("{} ".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))))
        print("[{}] ".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))), end="")
        ret = func(*args, **kwargs)
        return ret
    return wrapper


@add_time
def myprint(string, filename):
    with open(filename, "a") as f:
        f.write("{}\n".format(string))
    print(string)


def draw_two_dimension(
    y_lists,
    x_list,
    color_list,
    line_style_list,
    legend_list=None,
    legend_location="auto",
    legend_bbox_to_anchor=(0.515, 1.11),
    legend_ncol=3,
    legend_fontsize=15,
    fig_title=None,
    fig_x_label="time",
    fig_y_label="val",
    show_flag=True,
    save_flag=False,
    save_path=None,
    save_dpi=300,
    fig_title_size=20,
    fig_grid=False,
    marker_size=0,
    line_width=2,
    x_label_size=15,
    y_label_size=15,
    number_label_size=15,
    fig_size=(8, 6),
    x_ticks_set_flag=False,
    x_ticks=None,
    y_ticks_set_flag=False,
    y_ticks=None,
    tight_layout_flag=True,
    x_ticks_dress=None,
    y_ticks_dress=None
) -> None:
    """
    Draw a 2D plot of several lines
    :param y_lists: (list[list]) y value of lines, each list in which is one line. e.g., [[2,3,4,5], [2,1,0,-1], [1,4,9,16]]
    :param x_list: (list) x value shared by all lines. e.g., [1,2,3,4]
    :param color_list: (list) color of each line. e.g., ["red", "blue", "green"]
    :param line_style_list: (list) line style of each line. e.g., ["solid", "dotted", "dashed"]
    :param legend_list: (list) legend of each line, which CAN BE LESS THAN NUMBER of LINES. e.g., ["red line", "blue line", "green line"]
    :param legend_fontsize: (float) legend fontsize. e.g., 15
    :param fig_title: (string) title of the figure. e.g., "Anonymous"
    :param fig_x_label: (string) x label of the figure. e.g., "time"
    :param fig_y_label: (string) y label of the figure. e.g., "val"
    :param show_flag: (boolean) whether you want to show the figure. e.g., True
    :param save_flag: (boolean) whether you want to save the figure. e.g., False
    :param save_path: (string) If you want to save the figure, give the save path. e.g., "./test.png"
    :param save_dpi: (integer) If you want to save the figure, give the save dpi. e.g., 300
    :param fig_title_size: (float) figure title size. e.g., 20
    :param fig_grid: (boolean) whether you want to display the grid. e.g., True
    :param marker_size: (float) marker size. e.g., 0
    :param line_width: (float) line width. e.g., 1
    :param x_label_size: (float) x label size. e.g., 15
    :param y_label_size: (float) y label size. e.g., 15
    :param number_label_size: (float) number label size. e.g., 15
    :param fig_size: (tuple) figure size. e.g., (8, 6)
    :param x_ticks: (list) list of x_ticks. e.g., range(2, 21, 1)
    :param x_ticks_set_flag: (boolean) whether to set x_ticks. e.g., False
    :param y_ticks: (list) list of y_ticks. e.g., range(2, 21, 1)
    :param y_ticks_set_flag: (boolean) whether to set y_ticks. e.g., False
    :return:
    """
    assert len(list(y_lists[0])) == len(list(x_list)), "Dimension of y should be same to that of x"
    assert len(y_lists) == len(line_style_list) == len(color_list), "number of lines should be fixed"
    y_count = len(y_lists)
    plt.figure(figsize=fig_size)
    for i in range(y_count):
        plt.plot(x_list, y_lists[i], markersize=marker_size, linewidth=line_width, c=color_list[i], linestyle=line_style_list[i])
    plt.xlabel(fig_x_label, fontsize=x_label_size)
    plt.ylabel(fig_y_label, fontsize=y_label_size)
    if x_ticks_set_flag:
        if x_ticks_dress:
            plt.xticks(x_ticks, x_ticks_dress)
        else:
            plt.xticks(x_ticks)
    if y_ticks_set_flag:
        if y_ticks_dress:
            plt.xticks(y_ticks, y_ticks_dress)
        else:
            plt.yticks(y_ticks)
    plt.tick_params(labelsize=number_label_size)
    if legend_list:
        if legend_location == "fixed":
            plt.legend(legend_list, fontsize=legend_fontsize, bbox_to_anchor=legend_bbox_to_anchor, fancybox=True, ncol=legend_ncol)
        else:
            plt.legend(legend_list, fontsize=legend_fontsize)
    if fig_title:
        plt.title(fig_title, fontsize=fig_title_size)
    if fig_grid:
        plt.grid(True)
    if tight_layout_flag:
        plt.tight_layout()
    if save_flag:
        plt.savefig(save_path, dpi=save_dpi)
    if show_flag:
        plt.show()
    plt.clf()
    plt.close()


def draw_two_dimension_different_x(
    y_lists,
    x_lists,
    color_list,
    line_style_list,
    legend_list=None,
    legend_location="auto",
    legend_bbox_to_anchor=(0.515, 1.11),
    legend_ncol=3,
    legend_fontsize=15,
    fig_title=None,
    fig_x_label="time",
    fig_y_label="val",
    show_flag=True,
    save_flag=False,
    save_path=None,
    save_dpi=300,
    fig_title_size=20,
    fig_grid=False,
    marker_size=0,
    line_width=2,
    x_label_size=15,
    y_label_size=15,
    number_label_size=15,
    fig_size=(8, 6),
    x_ticks_set_flag=False,
    x_ticks=None,
    y_ticks_set_flag=False,
    y_ticks=None,
    tight_layout_flag=True
) -> None:
    """
    Draw a 2D plot of several lines
    :param y_lists: (list[list]) y value of lines, each list in which is one line. e.g., [[2,3,4,5], [2,1,0,-1], [1,4,9,16]]
    :param x_lists: (list[list]) x value of lines. e.g., [[1,2,3,4], [5,6,7]]
    :param color_list: (list) color of each line. e.g., ["red", "blue", "green"]
    :param line_style_list: (list) line style of each line. e.g., ["solid", "dotted", "dashed"]
    :param legend_list: (list) legend of each line, which CAN BE LESS THAN NUMBER of LINES. e.g., ["red line", "blue line", "green line"]
    :param legend_fontsize: (float) legend fontsize. e.g., 15
    :param fig_title: (string) title of the figure. e.g., "Anonymous"
    :param fig_x_label: (string) x label of the figure. e.g., "time"
    :param fig_y_label: (string) y label of the figure. e.g., "val"
    :param show_flag: (boolean) whether you want to show the figure. e.g., True
    :param save_flag: (boolean) whether you want to save the figure. e.g., False
    :param save_path: (string) If you want to save the figure, give the save path. e.g., "./test.png"
    :param save_dpi: (integer) If you want to save the figure, give the save dpi. e.g., 300
    :param fig_title_size: (float) figure title size. e.g., 20
    :param fig_grid: (boolean) whether you want to display the grid. e.g., True
    :param marker_size: (float) marker size. e.g., 0
    :param line_width: (float) line width. e.g., 1
    :param x_label_size: (float) x label size. e.g., 15
    :param y_label_size: (float) y label size. e.g., 15
    :param number_label_size: (float) number label size. e.g., 15
    :param fig_size: (tuple) figure size. e.g., (8, 6)
    :param x_ticks: (list) list of x_ticks. e.g., range(2, 21, 1)
    :param x_ticks_set_flag: (boolean) whether to set x_ticks. e.g., False
    :param y_ticks: (list) list of y_ticks. e.g., range(2, 21, 1)
    :param y_ticks_set_flag: (boolean) whether to set y_ticks. e.g., False
    :return:
    """
    assert len(y_lists) == len(line_style_list) == len(color_list), "number of lines should be fixed"
    y_count = len(y_lists)
    for i in range(y_count):
        assert len(y_lists[i]) == len(x_lists[i]), "Dimension of y should be same to that of x"
    plt.figure(figsize=fig_size)
    for i in range(y_count):
        plt.plot(x_lists[i], y_lists[i], markersize=marker_size, linewidth=line_width, c=color_list[i], linestyle=line_style_list[i])
    plt.xlabel(fig_x_label, fontsize=x_label_size)
    plt.ylabel(fig_y_label, fontsize=y_label_size)
    if x_ticks_set_flag:
        plt.xticks(x_ticks)
    if y_ticks_set_flag:
        plt.yticks(y_ticks)
    plt.tick_params(labelsize=number_label_size)
    if legend_list:
        if legend_location == "fixed":
            plt.legend(legend_list, fontsize=legend_fontsize, bbox_to_anchor=legend_bbox_to_anchor, fancybox=True, ncol=legend_ncol)
        else:
            plt.legend(legend_list, fontsize=legend_fontsize)
    if fig_title:
        plt.title(fig_title, fontsize=fig_title_size)
    if fig_grid:
        plt.grid(True)
    if tight_layout_flag:
        plt.tight_layout()
    if save_flag:
        plt.savefig(save_path, dpi=save_dpi)
    if show_flag:
        plt.show()
    plt.clf()
    plt.close()


def smooth_conv(data, kernel_size: int = 10):
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(data, kernel, mode='same')


def draw_multiple_loss(
        loss_path_list,
        color_list,
        line_style_list,
        legend_list,
        fig_title,
        start_index,
        end_index,
        threshold=None,
        smooth_kernel_size=1,
        marker_size=0,
        line_width=1,
        fig_size=(8, 6),
        x_ticks_set_flag=False,
        x_ticks=None,
        y_ticks_set_flag=False,
        y_ticks=None,
        show_flag=True,
        save_flag=False,
        save_path=None,
        only_original_flag=False,
        fig_x_label="epoch",
        fig_y_label="loss",
        legend_location="auto",
        legend_bbox_to_anchor=(0.515, 1.11),
        legend_ncol=3,
        tight_layout_flag=True
):
    # line_n = len(loss_path_list)
    assert (len(loss_path_list) if only_original_flag else 2 * len(loss_path_list)) == len(color_list) == len(line_style_list) == len(legend_list), "Note that for each loss in loss_path_list, this function will generate an original version and a smoothed version. So please give the color_list, line_style_list, legend_list for all of them"
    x_list = range(start_index, end_index)
    y_lists = [np.load(one_path) for one_path in loss_path_list]
    print("length:", [len(item) for item in y_lists])
    y_lists_smooth = [smooth_conv(item, smooth_kernel_size) for item in y_lists]
    for i, item in enumerate(y_lists):
        print("{}: {}".format(legend_list[i], np.mean(item[start_index: end_index])))
        if threshold:
            match_index_list = np.where(y_lists_smooth[i] <= threshold)
            if len(match_index_list[0]) == 0:
                print("No index of epoch matches condition '< {}'!".format(threshold))
            else:
                print("Epoch {} is the first value matches condition '< {}'!".format(match_index_list[0][0], threshold))
    y_lists = [item[x_list] for item in y_lists]
    y_lists_smooth = [item[x_list] for item in y_lists_smooth]
    draw_two_dimension(
        y_lists=y_lists if only_original_flag else y_lists + y_lists_smooth,
        x_list=x_list,
        color_list=color_list,
        legend_list=legend_list,
        legend_location=legend_location,
        legend_bbox_to_anchor=legend_bbox_to_anchor,
        legend_ncol=legend_ncol,
        line_style_list=line_style_list,
        fig_title=fig_title,
        fig_size=fig_size,
        fig_x_label=fig_x_label,
        fig_y_label=fig_y_label,
        x_ticks_set_flag=x_ticks_set_flag,
        y_ticks_set_flag=y_ticks_set_flag,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        marker_size=marker_size,
        line_width=line_width,
        show_flag=show_flag,
        save_flag=save_flag,
        save_path=save_path,
        tight_layout_flag=tight_layout_flag
    )


class MultiSubplotDraw:
    def __init__(self, row, col, fig_size=(8, 6), show_flag=True, save_flag=False, save_path=None, save_dpi=300, tight_layout_flag=False):
        self.row = row
        self.col = col
        self.subplot_index = 0
        self.show_flag = show_flag
        self.save_flag = save_flag
        self.save_path = save_path
        self.save_dpi = save_dpi
        self.tight_layout_flag = tight_layout_flag
        self.fig = plt.figure(figsize=fig_size)

    def draw(self, ):
        if self.tight_layout_flag:
            plt.tight_layout()
        if self.save_flag:
            plt.savefig(self.save_path, dpi=self.save_dpi)
        if self.show_flag:
            plt.show()
        plt.clf()
        plt.close()

    def add_subplot(
            self,
            y_lists,
            x_list,
            color_list,
            line_style_list,
            legend_list=None,
            legend_location="auto",
            legend_bbox_to_anchor=(0.515, 1.11),
            legend_ncol=3,
            legend_fontsize=15,
            fig_title=None,
            fig_x_label="time",
            fig_y_label="val",
            fig_title_size=20,
            fig_grid=False,
            marker_size=0,
            line_width=2,
            x_label_size=15,
            y_label_size=15,
            number_label_size=15,
            x_ticks_set_flag=False,
            x_ticks=None,
            y_ticks_set_flag=False,
            y_ticks=None,
    ) -> None:
        # assert len(list(y_lists[0])) == len(list(x_list)), "Dimension of y should be same to that of x"
        assert len(y_lists) == len(line_style_list) == len(color_list), "number of lines should be fixed"
        y_count = len(y_lists)
        self.subplot_index += 1
        ax = self.fig.add_subplot(self.row, self.col, self.subplot_index)
        for i in range(y_count):
            draw_length = min(len(x_list), len(y_lists[i]))
            ax.plot(x_list[:draw_length], y_lists[i][:draw_length], markersize=marker_size, linewidth=line_width, c=color_list[i], linestyle=line_style_list[i])
        ax.set_xlabel(fig_x_label, fontsize=x_label_size)
        ax.set_ylabel(fig_y_label, fontsize=y_label_size)
        if x_ticks_set_flag:
            ax.set_xticks(x_ticks)
        if y_ticks_set_flag:
            ax.set_yticks(y_ticks)
        if legend_list:
            if legend_location == "fixed":
                ax.legend(legend_list, fontsize=legend_fontsize, bbox_to_anchor=legend_bbox_to_anchor, fancybox=True,
                           ncol=legend_ncol)
            else:
                ax.legend(legend_list, fontsize=legend_fontsize)
        if fig_title:
            ax.set_title(fig_title, fontsize=fig_title_size)
        if fig_grid:
            ax.grid(True)
        plt.tick_params(labelsize=number_label_size)


if __name__ == "__main__":
    # x_list = range(10000)
    # y_lists = [
    #     [0.005 * i + 10 for i in x_list],
    #     [-0.005 * i - 30 for i in x_list],
    #     [0.008 * i - 10 for i in x_list],
    #     [-0.006 * i - 20 for i in x_list],
    #     [-0.001 * i - 5 for i in x_list],
    #     [-0.003 * i - 1 for i in x_list]
    # ]
    # color_list = ["red", "blue", "green", "cyan", "black", "purple"]
    # line_style_list = ["dashed", "dotted", "dashdot", "dashdot", "dashdot", "dashdot"]
    # legend_list = ["red line", "blue line", "green line", "cyan line", "black line", "purple line"]
    # #
    # draw_two_dimension(
    #     y_lists=y_lists,
    #     x_list=x_list,
    #     color_list=color_list,
    #     legend_list=legend_list,
    #     line_style_list=line_style_list,
    #     fig_title=None,
    #     fig_size=(8, 6),
    #     show_flag=True,
    #     save_flag=False,
    #     save_path=None,
    #     legend_location="fixed",
    #     legend_ncol=3,
    #     legend_bbox_to_anchor=(0.86, 1.19),
    #     tight_layout_flag=True
    # )

    x_list = range(10000)
    y_lists = [
        [0.005 * i + 10 for i in x_list],
        [-0.005 * i - 30 for i in x_list],
        # [0.008 * i - 10 for i in x_list],
        # [-0.006 * i - 20 for i in x_list],
        # [-0.001 * i - 5 for i in x_list],
        # [-0.003 * i - 1 for i in x_list]
    ]

    m = MultiSubplotDraw(row=3, col=1, tight_layout_flag=True)
    m.add_subplot(
        y_lists=y_lists,
        x_list=x_list,
        color_list=["r", "b"],
        legend_list=["111", "222"],
        line_style_list=["dashed", "solid"],
        fig_title="hello world")

    m.add_subplot(
        y_lists=y_lists,
        x_list=x_list,
        color_list=["yellow", "b"],
        legend_list=["111", "222"],
        line_style_list=["dashed", "solid"],
        fig_title="hello world")

    m.add_subplot(
        y_lists=y_lists,
        x_list=x_list,
        color_list=["g", "grey"],
        legend_list=["111", "222"],
        line_style_list=["dashed", "solid"],
        fig_title="hello world")

    m.draw()
    # color_list = ["red", "blue"]#, "green", "cyan", "black", "purple"]
    # line_style_list = ["dashed", "dotted"]#, "dashdot", "dashdot", "dashdot", "dashdot"]
    # legend_list = ["red line", "blue line"]#, "green line", "cyan line", "black line", "purple line"]
    # #
    # draw_two_dimension(
    #     y_lists=y_lists,
    #     x_list=x_list,
    #     color_list=color_list,
    #     legend_list=legend_list,
    #     line_style_list=line_style_list,
    #     fig_title=None,
    #     fig_size=(8, 6),
    #     show_flag=True,
    #     save_flag=False,
    #     save_path=None,
    #     legend_location="fixed",
    #     legend_ncol=3,
    #     legend_bbox_to_anchor=(0.515, 1.11),#(0.86, 1.19),
    #     tight_layout_flag=True
    # )

    # fig = plt.figure(figsize=(16, 6))
    # ax = fig.add_subplot(1, 3, 1)
    # ax.plot(x_list, [0.005 * i + 10 for i in x_list])
    # ax.legend(["aaa"])
    # ax.set_xlabel("hello", fontsize=20)
    # ax.set_xticks(range(0, 10001, 2000))
    # ax.grid(True)
    #
    # ax = fig.add_subplot(1, 3, 2)
    # ax.plot(x_list, [-0.005 * i - 30 for i in x_list])
    #
    # ax = fig.add_subplot(1, 3, 3)
    # ax.plot(x_list, [-0.005 * i - 30 for i in x_list])
    #
    # plt.tick_params(labelsize=20)
    # plt.show()
    # plt.close()

    # import numpy as np
    # x1 = [i * 0.01 for i in range(1000)]
    # y1 = [np.sin(i * 0.01) for i in range(1000)]
    # x2 = [i * 0.01 for i in range(500)]
    # y2 = [np.cos(i * 0.01) for i in range(500)]
    # x3 = []
    # y3 = []
    # draw_two_dimension_different_x(
    #     y_lists=[y1, y2, y3],
    #     x_lists=[x1, x2, x3],
    #     color_list=["b", "r", "y"],
    #     legend_list=["sin", "cos", "balabala"],
    #     line_style_list=["dotted", "dotted", "dotted"],
    #     fig_title="Anonymous",
    #     fig_size=(8, 6),
    #     show_flag=True,
    #     save_flag=False,
    #     save_path=None
    # )

    # data = np.load("loss/SimpleNetworkSIRAges_Truth_100000_1000_0.01_2022-06-05-20-54-55_loss_100000.npy")
    # print(type(data))
    # print(data)
    # data_10 = smooth_conv(data, 500)
    # print(data_10)
    # data_20 = smooth_conv(data, 1000)
    # print(data_20)
    # start_index = 40000
    # end_index = 100000
    #
    # draw_two_dimension(
    #     y_lists=[data[start_index: end_index], data_10[start_index: end_index], data_20[start_index: end_index]],
    #     x_list=range(start_index, end_index),
    #     color_list=["r", "g", "b"],
    #     legend_list=["None", "10", "20"],
    #     line_style_list=["solid"] * 3,
    #     fig_title="Anonymous",
    #     fig_size=(8, 6),
    #     show_flag=True,
    #     save_flag=False,
    #     save_path=None
    # )

    # a = np.asarray([1,2,4,5,7, 0])
    # print(np.where(a > 3))
