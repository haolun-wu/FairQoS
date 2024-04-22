import json
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams["figure.figsize"] = (30, 27)

# import seaborn as sns
# sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (24, 9)
plt.rcParams["font.size"] = 20
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rcParams['pdf.fonttype'] = 42
from matplotlib.pyplot import cm

# plt.rcParams['image.cmap']='copper'


# color_list = list(cm.RdPu(np.linspace(0, 1, 8)))
color_list = list(cm.GnBu(np.linspace(0, 1, 8)))


def parse_args():
    parser = ArgumentParser(description="InterFair")
    parser.add_argument('--data', type=str, default='ml-1m', choices=['ml-1m'],
                        help="File path for data")
    parser.add_argument('--model', type=str, default='BPRMF')
    # (100,1)
    parser.add_argument('--s_ep', type=int, default=100)
    parser.add_argument('--r_ep', type=int, default=1)

    # parser.add_argument('--norm', type=str, default='N')
    # parser.add_argument('--coll', type=str, default='Y')

    return parser.parse_args()


def NormalizeData(data):
    data = np.array(data)
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def plot_scatter(index, data1, data2, scale, x_name, y_name):
    data1 = np.array(data1)[index]
    data2 = np.array(data2)[index] * scale
    # data1 = (data1 - data1.min()) / (data1.max() - data1.min())
    # data2 = (data2 - data2.min()) / (data2.max() - data2.min())
    plt.scatter(data1, data2)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()


def axes_plot_scatter(index, data1, data2, scale, x_name, y_name, axes):
    data1 = np.array(data1)[index]
    data2 = np.array(data2)[index] * scale
    # data1 = (data1 - data1.min()) / (data1.max() - data1.min())
    # data2 = (data2 - data2.min()) / (data2.max() - data2.min())
    axes.scatter(data1, data2)
    axes.set_xlabel(x_name, fontsize=36)
    axes.set_ylabel(y_name, fontsize=36)
    axes.tick_params(labelsize=20)


def plot_box_trade_plus(metric_name, axes):
    with open("./save_exp/{}/{}F_all_{}.json".format(args.data, metric_name, args.model)) as f:
        data_F_raw = json.load(f)
    with open("./save_exp/{}/{}D_all_{}.json".format(args.data, metric_name, args.model)) as f:
        data_D_raw = json.load(f)
    with open("./save_exp/{}/{}R_all_{}.json".format(args.data, metric_name, args.model)) as f:
        data_R_raw = json.load(f)
    with open("./save_exp/{}/{}F_all_{}_static.json".format(args.data, metric_name, args.model)) as f:
        data_F_static = json.load(f)
    with open("./save_exp/{}/{}D_all_{}_static.json".format(args.data, metric_name, args.model)) as f:
        data_D_static = json.load(f)
    with open("./save_exp/{}/{}R_all_{}_static.json".format(args.data, metric_name, args.model)) as f:
        data_R_static = json.load(f)

    # The last one is the static
    data_F_raw.extend(data_F_static)
    data_D_raw.extend(data_D_static)
    data_R_raw.extend(data_R_static)

    # print("metric_name:", metric_name)

    data_F_raw, data_D_raw, data_R_raw = np.array(data_F_raw), np.array(data_D_raw), np.array(data_R_raw)
    # print("data_F_raw:", data_F_raw)
    # print("data_D_raw:", data_D_raw)
    # print("data_R_raw:", data_R_raw)

    # data_F_raw, data_D_raw, data_R_raw = np.abs(data_F_raw), np.abs(data_D_raw), np.abs(data_R_raw)
    data_F, data_D, data_R = NormalizeData(data_F_raw), NormalizeData(data_D_raw), NormalizeData(data_R_raw)
    # data_F, data_D, data_R = data_F_raw, data_D_raw, data_R_raw

    # print("data_F:", data_F)
    # print("data_D:", data_D)
    # print("data_R:", data_R)

    length = len(data_D) - 1
    factor_index = []
    # From large to small
    for i in range(len_tau):
        factor_index.append(np.arange(0, length, len_tau) + i)
    # factor_index.append(np.arange(0, length, 8) + 1)  # 4
    # factor_index.append(np.arange(0, length, 8) + 2)  # 2
    # factor_index.append(np.arange(0, length, 8) + 3)  # 1
    # factor_index.append(np.arange(0, length, 8) + 4)  # 1/2
    # factor_index.append(np.arange(0, length, 8) + 5)  # 1/4
    # factor_index.append(np.arange(0, length, 8) + 6)  # 1/8
    # factor_index.append(np.arange(0, length, 8) + 7)  # 1/16

    data_F_matrix = np.array([data_F[factor_index[i]] for i in range(len_tau)])
    data_D_matrix = np.array([data_D[factor_index[i]] for i in range(len_tau)])
    data_R_matrix = np.array([data_R[factor_index[i]] for i in range(len_tau)])

    print("1:", data_F_matrix[0])
    print("2:", data_F_matrix[1])
    print("3:", data_F_matrix[2])

    # bolxplot of F
    data_tmp = []
    for i in range(len_tau):
        data_tmp.append(data_F_matrix[i])
    data_tmp.append(data_F[-1])
    axes[0].plot(data_tmp, marker='o', color="royalblue", linewidth=2)
    axes[0].bar(np.arange(8), data_tmp, color=color_list)
    # axes[0].boxplot(
    #     [data_F_matrix[0], data_F_matrix[1], data_F_matrix[2], data_F_matrix[3], data_F_matrix[4], data_F_matrix[5],
    #      data_F_matrix[6], data_F_matrix[7], data_F[-1]])

    # bolxplot of D
    data_tmp = []
    for i in range(len_tau):
        data_tmp.append(data_D_matrix[i])
    data_tmp.append(data_D[-1])
    axes[1].plot(data_tmp, marker='o', color="royalblue", linewidth=2)
    axes[1].bar(np.arange(8), data_tmp, color=color_list)
    # axes[1].boxplot(
    #     [data_D_matrix[0], data_D_matrix[1], data_D_matrix[2], data_D_matrix[3], data_D_matrix[4], data_D_matrix[5],
    #      data_D_matrix[6], data_D_matrix[7], data_D[-1]])

    # boxplot of R
    data_tmp = []
    for i in range(len_tau):
        data_tmp.append(data_R_matrix[i])
    data_tmp.append(data_R[-1])
    axes[2].plot(data_tmp, marker='o', color="royalblue", linewidth=2)
    axes[2].bar(np.arange(8), data_tmp, color=color_list)
    # axes[2].boxplot(
    #     [data_R_matrix[0], data_R_matrix[1], data_R_matrix[2], data_R_matrix[3], data_R_matrix[4], data_R_matrix[5],
    #      data_R_matrix[6], data_R_matrix[7], data_R[-1]])

    # tradeoff curve between D and R
    # data_F_avg = NormalizeData(np.append(data_F_raw[-1], np.mean(data_F_raw[:-1].reshape(-1, 5), axis=0)))
    # data_D_avg = NormalizeData(np.append(np.mean(data_D_raw[:-1].reshape(-1, len_tau), axis=0), data_D_raw[-1]))
    # data_R_avg = NormalizeData(np.append(np.mean(data_R_raw[:-1].reshape(-1, len_tau), axis=0), data_R_raw[-1]))

    # axes[3].plot(data_D_avg, data_R_avg, marker='o')

    axes[0].set_title("{}-F".format(metric_name))
    axes[1].set_title("{}-D".format(metric_name))
    axes[2].set_title("{}-R".format(metric_name))
    # axes[3].set_title("D/R trade-off".format(metric_name), fontsize=16)

    # xticks=["tau=8", "tau=4", "tau=2", "tau=1", "tau=0.5", "static"]
    # axes[0].set_xticklabels(xticks)
    # axes[1].set_xticklabels(xticks)
    # axes[2].set_xticklabels(xticks)


if __name__ == '__main__':
    args = parse_args()

    fig, axes = plt.subplots(nrows=3, ncols=6, sharey='col', sharex='row')

    rand_tau_list = [8, 4, 2, 1, 0.5, 0.25, 0.125]
    len_tau = len(rand_tau_list)

    # plot_box_trade("IID", "IIR", [axes[0, 0], axes[1, 0], axes[2, 0]])
    # plot_box_trade("IGD", "IGR", [axes[0, 1], axes[1, 1], axes[2, 1]])
    # plot_box_trade("GID", "GIR", [axes[0, 2], axes[1, 2], axes[2, 2]])
    # plot_box_trade("GGD", "GGR", [axes[0, 3], axes[1, 3], axes[2, 3]])
    # plot_box_trade("AID", "AIR", [axes[0, 4], axes[1, 4], axes[2, 4]])
    # plot_box_trade("AGD", "AGR", [axes[0, 5], axes[1, 5], axes[2, 5]])

    plot_box_trade_plus("II", [axes[0, 0], axes[1, 0], axes[2, 0]])
    plot_box_trade_plus("IG", [axes[0, 1], axes[1, 1], axes[2, 1]])
    plot_box_trade_plus("GI", [axes[0, 2], axes[1, 2], axes[2, 2]])
    plot_box_trade_plus("GG", [axes[0, 3], axes[1, 3], axes[2, 3]])
    plot_box_trade_plus("AI", [axes[0, 4], axes[1, 4], axes[2, 4]])
    plot_box_trade_plus("AG", [axes[0, 5], axes[1, 5], axes[2, 5]])

    xticks = ["8", "4", "2", "1", "1/2", "1/4", "1/8", "ST"]
    # xticks = ["t=1", "t=2", "t=3", "t=4", "t=5", "static"]
    # axes[0].set_xticklabels(xticks)
    # axes[1].set_xticklabels(xticks)
    # axes[2].set_xticklabels(xticks)
    for i in [0, 1, 2]:
        for j in [0, 1, 2, 3, 4, 5]:
            # for j in [4, 5]:
            axes[i, j].set_xticks(np.arange(len_tau+1))
            axes[i, j].set_xticklabels(xticks)

    # plt.suptitle('Model:{}'.format(args.model), y=0.98, fontsize=20)
    plt.tight_layout()
    plt.savefig('./fig/RQ1.1.pdf')

    plt.show()
