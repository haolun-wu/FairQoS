import json
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.special import softmax

# import numpy as np
from sklearn import metrics

plt.rcParams['pdf.fonttype'] = 42

# plt.rcParams["figure.figsize"] = (30,5)
# plt.rcParams["font.size"] = 18
# plt.rc('xtick', labelsize=18)
# plt.rc('ytick', labelsize=18)

sns.set_theme(style="whitegrid")

font_size = 28
label_size = 24
legend_size = 16.8


def parse_args():
    parser = ArgumentParser(description="InterFair")
    parser.add_argument('--data', type=str, default='ml-1m', choices=['ml-1m'],
                        help="File path for data")
    parser.add_argument('--model', type=str, default='BPRMF')
    # (100,1)
    parser.add_argument('--s_ep', type=int, default=100)
    parser.add_argument('--r_ep', type=int, default=1)

    parser.add_argument('--norm', type=str, default='N')
    parser.add_argument('--coll', type=str, default='Y')

    return parser.parse_args()


def NormalizeData(data):
    data = np.array(data)
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def NormalizeData_zero(data):
    data = np.array(data)
    return (data - np.min(data)) / (np.max(data) - np.min(data)) * np.max(data)


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


# def plot_tradeoff():
#     with open("./save_exp/{}/IID_all_{}.json".format(args.data, args.model)) as f:
#         IID = json.load(f)
#     with open("./save_exp/{}/IGD_all_{}.json".format(args.data, args.model)) as f:
#         IGD = json.load(f)
#     with open("./save_exp/{}/GID_all_{}.json".format(args.data, args.model)) as f:
#         GID = json.load(f)
#     with open("./save_exp/{}/GGD_all_{}.json".format(args.data, args.model)) as f:
#         GGD = json.load(f)
#     with open("./save_exp/{}/AID_all_{}.json".format(args.data, args.model)) as f:
#         AID = json.load(f)
#     with open("./save_exp/{}/AGD_all_{}.json".format(args.data, args.model)) as f:
#         AGD = json.load(f)
#
#     with open("./save_exp/{}/IIR_all_{}.json".format(args.data, args.model)) as f:
#         IIR = json.load(f)
#     with open("./save_exp/{}/IGR_all_{}.json".format(args.data, args.model)) as f:
#         IGR = json.load(f)
#     with open("./save_exp/{}/GIR_all_{}.json".format(args.data, args.model)) as f:
#         GIR = json.load(f)
#     with open("./save_exp/{}/GGR_all_{}.json".format(args.data, args.model)) as f:
#         GGR = json.load(f)
#     with open("./save_exp/{}/AIR_all_{}.json".format(args.data, args.model)) as f:
#         AIR = json.load(f)
#     with open("./save_exp/{}/AGR_all_{}.json".format(args.data, args.model)) as f:
#         AGR = json.load(f)
#
#     length = len(IID)
#
#     IID, IIR = NormalizeData(IID), NormalizeData(IIR)
#     IGD, IGR = NormalizeData(IGD), NormalizeData(IGR)
#     GID, GIR = NormalizeData(GID), NormalizeData(GIR)
#     GGD, GGR = NormalizeData(GGD), NormalizeData(GGR)
#     AID, AIR = NormalizeData(AID), NormalizeData(AIR)
#     AGD, AGR = NormalizeData(AGD), NormalizeData(AGR)
#
#     fig, axs = plt.subplots(6, 1)
#
#     axs[0].plot(IID, IIR, marker='o')
#     axs[1].plot(IGD, IGR, marker='o')
#     axs[2].plot(GID, GIR, marker='o')
#     axs[3].plot(GGD, GGR, marker='o')
#     axs[4].plot(AID, AIR, marker='o')
#     axs[5].plot(AGD, AGR, marker='o')
#
#     name_x = ["IID", "IGD", "GID", "GGD", "AID", "AGD"]
#     name_y = ["IIR", "IGR", "GIR", "GGR", "AIR", "AGR"]
#     for i in range(6):
#         axs[i].set_xlabel(name_x[i], fontsize=36)
#         axs[i].set_ylabel(name_y[i], fontsize=36)
#         axs[i].tick_params(labelsize=20)
#
#     axs[0].set_title("{}, {}".format(args.data, args.model), fontsize=40)
#
#     # plt.suptitle('Data:{}, model:{}, metric:{}, train 500 epochs'.format(args.data, args.model, value), y=0.99,
#     #              fontsize=42)
#
#     plt.tight_layout()
#     plt.show()
#     # fig.savefig("../fig/tradeoff_{}_{}.png".format(args.data, args.model), bbox_inches='tight')


# def plot_box_trade_plus(axes):
#     data_F_raw, data_F_static = [[], [], [], [], [], []], [[], [], [], [], [], []]
#     data_D_raw, data_D_static = [[], [], [], [], [], []], [[], [], [], [], [], []]
#     data_R_raw, data_R_static = [[], [], [], [], [], []], [[], [], [], [], [], []]
#     metric_name = ["II", "IG", "GI", "GG", "AI", "AG"]
#     for i in range(6):
#         with open("./save_exp/{}/{}F_all_{}.json".format(args.data, metric_name[i], args.model)) as f:
#             data_F_raw[i] = json.load(f)
#         with open("./save_exp/{}/{}D_all_{}.json".format(args.data, metric_name[i], args.model)) as f:
#             data_D_raw[i] = json.load(f)
#         with open("./save_exp/{}/{}R_all_{}.json".format(args.data, metric_name[i], args.model)) as f:
#             data_R_raw[i] = json.load(f)
#         with open("./save_exp/{}/{}F_all_{}_static.json".format(args.data, metric_name[i], args.model)) as f:
#             data_F_static[i] = json.load(f)
#         with open("./save_exp/{}/{}D_all_{}_static.json".format(args.data, metric_name[i], args.model)) as f:
#             data_D_static[i] = json.load(f)
#         with open("./save_exp/{}/{}R_all_{}_static.json".format(args.data, metric_name[i], args.model)) as f:
#             data_R_static[i] = json.load(f)
#
#     # The last one is the static
#     for i in range(6):
#         data_F_raw[i].extend(data_F_static[i])
#         data_D_raw[i].extend(data_D_static[i])
#         data_R_raw[i].extend(data_R_static[i])
#         data_F_raw[i] = np.array((data_F_raw[i]))
#         data_D_raw[i] = np.array((data_D_raw[i]))
#         data_R_raw[i] = np.array((data_R_raw[i]))
#
#     data_D_avg = [[], [], [], [], [], []]
#     data_R_avg = [[], [], [], [], [], []]
#
#     for i in range(6):
#         data_D_avg[i] = np.append(np.mean(data_D_raw[i][:-1].reshape(-1, len_tau), axis=0), data_D_raw[i][-1])
#         data_R_avg[i] = np.append(np.mean(data_R_raw[i][:-1].reshape(-1, len_tau), axis=0), data_R_raw[i][-1])
#         # data_D_avg[i] = NormalizeData(data_D_avg[i])
#         # data_R_avg[i] = NormalizeData(data_R_avg[i])
#
#         axes[i].plot(data_D_avg[i], data_R_avg[i], label=args.model)
#         axes[i].legend(loc='lower right', fontsize=legend_size)
#
#     # axes[3].set_title("D/R trade-off".format(metric_name), fontsize=16)
#
#     # xticks=["tau=8", "tau=4", "tau=2", "tau=1", "tau=0.5", "static"]
#     # axes[0].set_xticklabels(xticks)
#     # axes[1].set_xticklabels(xticks)
#     # axes[2].set_xticklabels(xticks)


if __name__ == '__main__':
    args = parse_args()

    fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(30, 5))

    rand_tau_list = [8, 4, 2, 1, 0.5, 0.25, 0.125]
    len_tau = len(rand_tau_list)

    data_D_raw, data_D_static = [[], [], [], [], [], []], [[], [], [], [], [], []]
    data_R_raw, data_R_static = [[], [], [], [], [], []], [[], [], [], [], [], []]

    data_D_res = [[], [], [], [], [], []]
    data_R_res = [[], [], [], [], [], []]

    # for model_name in ['BPRMF', 'LDA', 'WRMF', 'PureSVD', 'SLIM', 'RW', 'Pop']:
    model_list = [
        'BPRMF',
        'LDA',
        'PureSVD',
        'SLIM',
        'WRMF',
        # 'CHI2',
        # 'KLD',
        #
        # 'LMWI',
        # 'LMWU',
        # 'PLSA',
        # 'Pop',
        #
        # 'RM1',
        # 'RM2',
        # 'RSV',
        # 'RW',

        # 'UIR',

    ]
    for model_name in model_list:
        # print("model_name:", model_name)
        args.model = model_name

        metric_name = ["II", "IG", "GI", "GG", "AI", "AG"]
        for i in range(6):
            with open("./save_exp/{}/{}D_all_{}.json".format(args.data, metric_name[i], args.model)) as f:
                data_D_raw[i] = json.load(f)
            with open("./save_exp/{}/{}R_all_{}.json".format(args.data, metric_name[i], args.model)) as f:
                data_R_raw[i] = json.load(f)
            with open("./save_exp/{}/{}D_all_{}_static.json".format(args.data, metric_name[i], args.model)) as f:
                data_D_static[i] = json.load(f)
            with open("./save_exp/{}/{}R_all_{}_static.json".format(args.data, metric_name[i], args.model)) as f:
                data_R_static[i] = json.load(f)

        # The last one is the static
        for i in range(6):
            data_D_raw[i].extend(data_D_static[i])
            data_R_raw[i].extend(data_R_static[i])

            # data_D_raw[i] = np.array((data_D_raw[i]))
            # data_R_raw[i] = np.array((data_R_raw[i]))
            data_D_res[i].extend(NormalizeData_zero(data_D_raw[i]))
            data_R_res[i].extend(NormalizeData_zero(data_R_raw[i]))

    for i in range(6):
        data_D_res[i] = NormalizeData(np.array(data_D_res[i]))
        data_R_res[i] = NormalizeData(np.array(data_R_res[i]))

    for i in range(6):
        data_x = data_D_res[i].reshape(-1, 8)
        data_y = data_R_res[i].reshape(-1, 8)
        count_line = data_x.shape[0]
        # f = []
        # f.append(interp1d(data_x[0], data_y[0]))
        # f.append(interp1d(data_x[1], data_y[1]))
        # f.append(interp1d(data_x[2], data_y[2]))
        # f.append(interp1d(data_x[3], data_y[3]))
        # f.append(interp1d(data_x[4], data_y[4]))
        # min_x = np.array(data_x[:, -1]).min()
        #
        # auc_res = []
        # for j in range(len(data_x)):
        #     end_index = np.where(np.array(data_x[j]) < min_x)[-1][-1]
        #     inter_y = f[j](min_x)
        #     data_x_input = list(data_x[j][:end_index])
        #     data_x_input.append(min_x)
        #     data_y_input = list(data_y[j][:end_index])
        #     data_y_input.append(inter_y)
        #
        #     auc_res.append(metrics.auc(data_x_input, data_y_input))
        # print(auc_res)

        # print("i:", i)
        # print("data_x:", data_x)
        # print("data_y", data_y)
        # print("count_line:", count_line)

        for j in range(count_line):
            axes[i].plot(data_x[j], data_y[j], label=model_list[j], linewidth=2.8)
            axes[i].legend(loc="lower right", fontsize=legend_size)

    # print("data_D_res[0]:", data_D_res[0].reshape(-1, 8))
    # print("data_R_res[0][0]:", data_R_res[0].reshape(-1, 8)[0])

    axes[0].plot(data_D_res[0].reshape(-1, 8)[0], data_R_res[0].reshape(-1, 8)[0], label=model_list[0])
    axes[0].plot(data_D_res[0].reshape(-1, 8)[1], data_R_res[0].reshape(-1, 8)[1], label=model_list[1])
    axes[0].plot(data_D_res[0].reshape(-1, 8)[2], data_R_res[0].reshape(-1, 8)[2], label=model_list[2])
    axes[0].plot(data_D_res[0].reshape(-1, 8)[3], data_R_res[0].reshape(-1, 8)[3], label=model_list[3])

    # print("data_D_res:", data_D_res)

    xticks = ["8", "4", "2", "1", "1/2", "1/4", "1/8", "static"]

    x = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    y = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    axes[0].set_xlabel('II-D', fontsize=font_size)
    axes[0].set_ylabel('II-R', fontsize=font_size)

    axes[1].set_xlabel('IG-D', fontsize=font_size)
    axes[1].set_ylabel('IG-R', fontsize=font_size)

    axes[2].set_xlabel('GI-D', fontsize=font_size)
    axes[2].set_ylabel('GI-R', fontsize=font_size)

    axes[3].set_xlabel('GG-D', fontsize=font_size)
    axes[3].set_ylabel('GG-R', fontsize=font_size)

    axes[4].set_xlabel('AI-D', fontsize=font_size)
    axes[4].set_ylabel('AI-R', fontsize=font_size)

    axes[5].set_xlabel('AG-D', fontsize=font_size)
    axes[5].set_ylabel('AG-R', fontsize=font_size)

    for i in range(6):
        axes[i].set_xticks(x)
        axes[i].set_yticks(y)
        axes[i].tick_params(axis="x", labelsize=label_size)
        axes[i].tick_params(axis="y", labelsize=label_size)

    plt.tight_layout()
    plt.savefig('./fig/RQ1.2.pdf')
    plt.show()
