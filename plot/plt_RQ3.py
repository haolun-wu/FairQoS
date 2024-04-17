import json
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# plt.rcParams["figure.figsize"] = (30, 27)

sns.set_context("paper", rc={"font.size": 12, "axes.titlesize": 24, "axes.labelsize": 24})
plt.rcParams['pdf.fonttype'] = 42

# plt.rcParams["figure.figsize"] = (10, 10)
# plt.rcParams["font.size"] = 16
# plt.rc('xtick', labelsize=10)
# plt.rc('ytick', labelsize=10)


def parse_args():
    parser = ArgumentParser(description="InterFair")
    parser.add_argument('--data', type=str, default='ml-1m', choices=['ml-1m'],
                        help="File path for data")
    parser.add_argument('--model', type=str, default='LDA')
    parser.add_argument('--s_ep', type=int, default=100)
    parser.add_argument('--r_ep', type=int, default=1)

    parser.add_argument('--norm', type=str, default='N')
    parser.add_argument('--coll', type=str, default='Y')

    parser.add_argument('--normalize', type=str, default='N')
    parser.add_argument('--age', type=str, default='Y')

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


def plot_box_trade_plus(metric_name1, metric_name2, axes):
    with open("./save_exp/{}/{}F_all_{}_{}_{}.json".format(args.data, metric_name1, args.model, args.s_ep,
                                                           args.r_ep)) as f:
        data1 = json.load(f)
    with open("./save_exp/{}/{}F_all_{}_{}_{}.json".format(args.data, metric_name2, args.model, args.s_ep,
                                                           args.r_ep)) as f:
        data2 = json.load(f)

    # The last one is the static
    data_F_raw.extend(data_F_static)
    data_D_raw.extend(data_D_static)
    data_R_raw.extend(data_R_static)

    data_F_raw, data_D_raw, data_R_raw = np.array(data_F_raw), np.array(data_D_raw), np.array(data_R_raw)

    data_F, data_D, data_R = NormalizeData(data_F_raw), NormalizeData(data_D_raw), NormalizeData(data_R_raw)

    length = len(data_D) - 1
    factor_index = []  # 0.5, 1, 2, 4, 8
    factor_index.append(np.arange(0, length, 5))
    factor_index.append(np.arange(0, length, 5) + 1)
    factor_index.append(np.arange(0, length, 5) + 2)
    factor_index.append(np.arange(0, length, 5) + 3)
    factor_index.append(np.arange(0, length, 5) + 4)

    data_F_matrix = np.array([data_F[factor_index[0]], data_F[factor_index[1]], data_F[factor_index[2]],
                              data_F[factor_index[3]], data_F[factor_index[4]]])
    data_D_matrix = np.array([data_D[factor_index[0]], data_D[factor_index[1]], data_D[factor_index[2]],
                              data_D[factor_index[3]], data_D[factor_index[4]]])
    data_R_matrix = np.array([data_R[factor_index[0]], data_R[factor_index[1]], data_R[factor_index[2]],
                              data_R[factor_index[3]], data_R[factor_index[4]]])

    # bolxplot of F
    axes[0].boxplot(
        [data_F_matrix[4], data_F_matrix[3], data_F_matrix[2], data_F_matrix[1], data_F_matrix[0], data_F[-1]],
        showfliers=True, showcaps=True)

    # bolxplot of D
    axes[1].boxplot(
        [data_D_matrix[4], data_D_matrix[3], data_D_matrix[2], data_D_matrix[1], data_D_matrix[0], data_D[-1]],
        showfliers=True, showcaps=True)

    # boxplot of R
    axes[2].boxplot(
        [data_R_matrix[4], data_R_matrix[3], data_R_matrix[2], data_R_matrix[1], data_R_matrix[0], data_R[-1]],
        showfliers=True, showcaps=True)

    # tradeoff curve between D and R
    # data_F_avg = NormalizeData(np.append(data_F_raw[-1], np.mean(data_F_raw[:-1].reshape(-1, 5), axis=0)))
    data_D_avg = NormalizeData(np.append(data_D_raw[-1], np.mean(data_D_raw[:-1].reshape(-1, 5), axis=0)))
    data_R_avg = NormalizeData(np.append(data_R_raw[-1], np.mean(data_R_raw[:-1].reshape(-1, 5), axis=0)))

    axes[3].plot(data_D_avg, data_R_avg, marker='o')

    axes[0].set_title("{}-F".format(metric_name), fontsize=16)
    axes[1].set_title("{}-D".format(metric_name), fontsize=16)
    axes[2].set_title("{}-R".format(metric_name), fontsize=16)
    axes[3].set_title("D/R trade-off".format(metric_name), fontsize=16)

    # xticks=["tau=8", "tau=4", "tau=2", "tau=1", "tau=0.5", "static"]
    # axes[0].set_xticklabels(xticks)
    # axes[1].set_xticklabels(xticks)
    # axes[2].set_xticklabels(xticks)


if __name__ == '__main__':
    args = parse_args()
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))

    args.age = 'N'
    IID, IGD, GID, GGD, AID, AGD = [], [], [], [], [], []
    IIR, IGR, GIR, GGR, AIR, AGR = [], [], [], [], [], []
    IIF, IGF, GIF, GGF, AIF, AGF = [], [], [], [], [], []
    dict = {"IIF": IIF, "IGF": IGF, "GIF": GIF, "GGF": GGF, "AIF": AIF, "AGF": AGF,
            "IID": IID, "IGD": IGD, "GID": GID, "GGD": GGD, "AID": AID, "AGD": AGD,
            "IIR": IIR, "IGR": IGR, "GIR": GIR, "GGR": GGR, "AIR": AIR, "AGR": AGR}

    for model_name in [
        'BPRMF',
        'LDA',
        'Pop',
        'PureSVD',
        'SLIM',
        'WRMF',
        'CHI2',
        'KLD',

        'LMWI',
        'LMWU',
        'PLSA',

        'RM1',
        'RM2',
        'RSV',
        'RW',

        'UIR',

    ]:
        print("model_name:", model_name)
        args.model = model_name

        for key in dict:
            if args.age == 'Y':
                with open("./save_exp/{}/{}_all_{}_Y.json".format(args.data, key, args.model)) as f:
                    dict[key].extend(json.load(f))
            else:
                with open("./save_exp/{}/{}_all_{}.json".format(args.data, key, args.model)) as f:
                    dict[key].extend(json.load(f))

    # print("IID:", IID)

    import scipy

    x = [IID, IGD, GID, GGD, AID, AGD]
    y = [IIR, IGR, GIR, GGR, AIR, AGR]
    z = [IIF, IGF, GIF, GGF, AIF, AGF]

    kandal_D = np.zeros((6, 6))
    kandal_R = np.zeros((6, 6))
    kandal_F = np.zeros((6, 6))

    for i in range(6):
        for j in range(6):
            kandal_D[i][j] = scipy.stats.kendalltau(x[i], x[j]).correlation
            kandal_R[i][j] = scipy.stats.kendalltau(y[i], y[j]).correlation
            kandal_F[i][j] = scipy.stats.kendalltau(z[i], z[j]).correlation

    corr_D = np.corrcoef(np.array([IID, IGD, GID, GGD, AID, AGD]))
    corr_R = np.corrcoef(np.array([IIR, IGR, GIR, GGR, AIR, AGR]))
    corr_F = np.corrcoef(np.array([IIF, IGF, GIF, GGF, AIF, AGF]))
    labels_D = ["II-D", "IG-D", "GI-D", "GG-D", "AI-D", "AG-D"]
    labels_R = ["II-R", "IG-R", "GI-R", "GG-R", "AI-R", "AG-R"]
    labels_F = ["II-F", "IG-F", "GI-F", "GG-F", "AI-F", "AG-F"]

    #
    sns.heatmap(ax=axes[0, 0], data=kandal_F, cmap="YlGnBu", linewidths=0.5, annot=True, fmt='.3f')
    axes[0, 0].set_xticks(np.arange(0, 6) + 0.5)
    axes[0, 0].set_xticklabels(labels_F, fontsize=17)
    axes[0, 0].set_yticks(np.arange(0, 6) + 0.5)
    axes[0, 0].set_yticklabels(labels_F, fontsize=17)

    sns.heatmap(ax=axes[1, 0], data=kandal_D, cmap="YlGnBu", linewidths=0.5, annot=True, fmt='.3f')
    axes[1, 0].set_xticks(np.arange(0, 6) + 0.5)
    axes[1, 0].set_xticklabels(labels_D, fontsize=17)
    axes[1, 0].set_yticks(np.arange(0, 6) + 0.5)
    axes[1, 0].set_yticklabels(labels_D, fontsize=17)

    sns.heatmap(ax=axes[2, 0], data=kandal_R, cmap="YlGnBu", linewidths=0.5, annot=True, fmt='.3f')
    axes[2, 0].set_xticks(np.arange(0, 6) + 0.5)
    axes[2, 0].set_xticklabels(labels_R, fontsize=17)
    axes[2, 0].set_yticks(np.arange(0, 6) + 0.5)
    axes[2, 0].set_yticklabels(labels_R, fontsize=17)

    args.age = 'Y'
    IID, IGD, GID, GGD, AID, AGD = [], [], [], [], [], []
    IIR, IGR, GIR, GGR, AIR, AGR = [], [], [], [], [], []
    IIF, IGF, GIF, GGF, AIF, AGF = [], [], [], [], [], []
    dict = {"IIF": IIF, "IGF": IGF, "GIF": GIF, "GGF": GGF, "AIF": AIF, "AGF": AGF,
            "IID": IID, "IGD": IGD, "GID": GID, "GGD": GGD, "AID": AID, "AGD": AGD,
            "IIR": IIR, "IGR": IGR, "GIR": GIR, "GGR": GGR, "AIR": AIR, "AGR": AGR}

    for model_name in [
        'BPRMF',
        'LDA',
        'Pop',
        'PureSVD',
        'SLIM',
        'WRMF',
        'CHI2',
        'KLD',

        'LMWI',
        'LMWU',
        'PLSA',

        'RM1',
        'RM2',
        'RSV',
        'RW',

        'UIR',

    ]:
        print("model_name:", model_name)
        args.model = model_name

        for key in dict:
            if args.age == 'Y':
                with open("./save_exp/{}/{}_all_{}_Y.json".format(args.data, key, args.model)) as f:
                    dict[key].extend(json.load(f))
            else:
                with open("./save_exp/{}/{}_all_{}.json".format(args.data, key, args.model)) as f:
                    dict[key].extend(json.load(f))

    # print("IID:", IID)

    import scipy

    x = [IID, IGD, GID, GGD, AID, AGD]
    y = [IIR, IGR, GIR, GGR, AIR, AGR]
    z = [IIF, IGF, GIF, GGF, AIF, AGF]

    kandal_D = np.zeros((6, 6))
    kandal_R = np.zeros((6, 6))
    kandal_F = np.zeros((6, 6))

    for i in range(6):
        for j in range(6):
            kandal_D[i][j] = scipy.stats.kendalltau(x[i], x[j]).correlation
            kandal_R[i][j] = scipy.stats.kendalltau(y[i], y[j]).correlation
            kandal_F[i][j] = scipy.stats.kendalltau(z[i], z[j]).correlation

    corr_D = np.corrcoef(np.array([IID, IGD, GID, GGD, AID, AGD]))
    corr_R = np.corrcoef(np.array([IIR, IGR, GIR, GGR, AIR, AGR]))
    corr_F = np.corrcoef(np.array([IIF, IGF, GIF, GGF, AIF, AGF]))
    labels_D = ["II-D", "IG-D", "GI-D", "GG-D", "AI-D", "AG-D"]
    labels_R = ["II-R", "IG-R", "GI-R", "GG-R", "AI-R", "AG-R"]
    labels_F = ["II-F", "IG-F", "GI-F", "GG-F", "AI-F", "AG-F"]

    #
    sns.heatmap(ax=axes[0, 1], data=kandal_F, cmap="YlGnBu", linewidths=0.5, annot=True, fmt='.3f')
    axes[0, 1].set_xticks(np.arange(0, 6) + 0.5)
    axes[0, 1].set_xticklabels(labels_F, fontsize=17)
    axes[0, 1].set_yticks(np.arange(0, 6) + 0.5)
    axes[0, 1].set_yticklabels(labels_F, fontsize=17)

    sns.heatmap(ax=axes[1, 1], data=kandal_D, cmap="YlGnBu", linewidths=0.5, annot=True, fmt='.3f')
    axes[1, 1].set_xticks(np.arange(0, 6) + 0.5)
    axes[1, 1].set_xticklabels(labels_D, fontsize=17)
    axes[1, 1].set_yticks(np.arange(0, 6) + 0.5)
    axes[1, 1].set_yticklabels(labels_D, fontsize=17)

    sns.heatmap(ax=axes[2, 1], data=kandal_R, cmap="YlGnBu", linewidths=0.5, annot=True, fmt='.3f')
    axes[2, 1].set_xticks(np.arange(0, 6) + 0.5)
    axes[2, 1].set_xticklabels(labels_R, fontsize=17)
    axes[2, 1].set_yticks(np.arange(0, 6) + 0.5)
    axes[2, 1].set_yticklabels(labels_R, fontsize=17)

    axes[2, 0].set_title('(a) user: gender / item: genre', fontsize=18, y=-0.1, pad=-25, fontweight="bold")
    axes[2, 1].set_title('(b) user: age / item: genre', fontsize=18, y=-0.1, pad=-25, fontweight="bold")
    #

    #
    #
    #
    plt.tight_layout()
    plt.savefig('./fig/RQ2.pdf')
    plt.show()
