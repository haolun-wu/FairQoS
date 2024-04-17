import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams['pdf.fonttype'] = 42
legend_size = 16.59
marker_size = 14
line_width = 3

label_size = 18
font_size = 20

m_size = 10
l_width = 3


def NormalizeData(data):
    data = np.array(data)
    return (data - np.min(data)) / (np.max(data) - np.min(data))


if __name__ == '__main__':
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    # x = np.array([0.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]) * 1e-3
    # y = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])*1e-6

    # ax.set_title('(a) Amazon_CD', fontsize=36, y=-0.1, pad=-75, fontname="Times New Roman", fontweight="bold")

    data_II = np.array([1.1076, 1.1103, 1.1173, 1.1633, 1.2245, 1.4235])  # * 1e-3
    data_GG = np.array([1.9835, 1.6723, 1.4328, 1.3076, 1.2235, 1.0235])  # * 1e-6
    # data_II = NormalizeData(data_II)
    # data_GG = NormalizeData(data_GG)
    axes[0].plot(data_II, data_GG, marker="o", markersize=m_size, linewidth=l_width)
    # axes[0].set_xticks(x)
    # axes[0].set_yticks(y)
    axes[0].set_ylim(ymin=0, ymax=2.2)
    axes[0].set_xlim(xmin=0, xmax=1.7)
    axes[0].tick_params(axis="x", labelsize=label_size)
    axes[0].tick_params(axis="y", labelsize=label_size)
    axes[0].set_ylabel('GG-F (x 1e-6)', fontsize=font_size)
    axes[0].set_xlabel('II-F (x 1e-3)', fontsize=font_size)
    axes[0].set_title('(a) MovieLens100k', fontsize=font_size, y=-0.1, pad=-45, fontweight="bold")

    data_II = np.array([1.2276, 1.2703, 1.3073, 1.4033, 1.5345, 1.82235])  # * 1e-3
    data_GG = np.array([1.9835, 1.6723, 1.3328, 1.1776, 1.0735, 1.0005])  # * 1e-6
    # data_II = NormalizeData(data_II)
    # data_GG = NormalizeData(data_GG)
    axes[1].plot(data_II, data_GG, marker="o", markersize=m_size, linewidth=l_width)
    # axes[1].set_xticks(x)
    # axes[1].set_yticks(y)
    axes[1].set_ylim(ymin=0, ymax=2.2)
    axes[1].set_xlim(xmin=0, xmax=2.3)
    axes[1].tick_params(axis="x", labelsize=label_size)
    axes[1].tick_params(axis="y", labelsize=label_size)
    axes[1].set_ylabel('GG-F (x 1e-6)', fontsize=font_size)
    axes[1].set_xlabel('II-F (x 1e-3)', fontsize=font_size)
    axes[1].set_title('(b) MovieLens1M', fontsize=font_size, y=-0.1, pad=-45, fontweight="bold")

    plt.tight_layout()
    plt.savefig('./fig/plot_OPT.pdf')
    plt.show()
