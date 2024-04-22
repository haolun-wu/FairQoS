import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# import seaborn as sns
# sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (22, 9)
plt.rcParams["font.size"] = 20
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rcParams['pdf.fonttype'] = 42
from matplotlib.pyplot import cm

cmap = sns.cubehelix_palette(as_cmap=True)
color_list_mpc_mpv = list(cmap(np.linspace(0, 1, 32)))[8]
color_list_gmpc_gmpv = list(cmap(np.linspace(0, 1, 32)))[20]
# color_list_mpc_mpv = list(cm.GnBu(np.linspace(0, 1, 32)))[19]


# Mock data setup - replace with your actual data
len_tau = 8  # Number of data points minus the last one, which is added separately
value_MPC = [
    # First row
    [0, 0.05, 0.35, 0.62, 0.93, 0.81, 0.77, 0.64],
    [0, 0.02, 0.21, 0.38, 0.57, 0.53, 0.44, 0.31],
    [0, 0.04, 0.23, 0.39, 0.44, 0.39, 0.29, 0.16],
    [0, 0.05, 0.29, 0.43, 0.46, 0.34, 0.22, 0.18]]
value_GMPC = [
    [0, 0.1, 0.4, 0.7, 1, 0.9, 0.83, 0.72],
    [0, 0.03, 0.41, 0.68, 1, 0.93, 0.79, 0.70],
    [0, 0.12, 0.53, 0.79, 0.94, 1, 0.89, 0.81],
    [0, 0.15, 0.59, 0.83, 0.96, 1, 0.92, 0.78]]
value_MPV = [
    [0, 0.02, 0.40, 0.58, 0.90, 0.88, 0.80, 0.62],
    [0, 0.03, 0.21, 0.41, 0.50, 0.52, 0.55, 0.46],
    [0, 0.12, 0.20, 0.30, 0.35, 0.4, 0.39, 0.33],
    [0, 0.10, 0.21, 0.30, 0.36, 0.4, 0.32, 0.30]]
value_GMPV = [
    [0, 0.06, 0.42, 0.68, 1, 0.92, 0.85, 0.7],
    [0, 0.09, 0.41, 0.71, 1, 0.94, 0.88, 0.75],
    [0, 0.22, 0.60, 0.81, 0.95, 1, 0.89, 0.79],
    [0, 0.25, 0.61, 0.85, 0.96, 1, 0.92, 0.80],
]  # Replace with your actual data

fig, axes = plt.subplots(nrows=2, ncols=4, sharey='col', sharex='row')

# Titles for the subplots
titles = ["DA-SS", "GA-SS", r'GA-SS$_{\sum \prod}$', r'GA-SS$_{\prod \sum}$']
xticks = ["8", "4", "2", "1", "1/2", "1/4", "1/8", "Static"]
bar_width = 0.36  # Adjust as needed
# Set positions for the groups of bars
indices = np.arange(len(xticks))

# Plot side-by-side bins and curves
# Plotting loop
for i, (values1, values2, colors1, colors2) in enumerate(zip(
        [value_MPC, value_MPV], [value_GMPC, value_GMPV],
        [color_list_mpc_mpv, color_list_mpc_mpv], [color_list_gmpc_gmpv, color_list_gmpc_gmpv])):
    for j, ax in enumerate(axes[i]):
        # Bar positions
        bar_positions_1 = indices - bar_width / 2
        bar_positions_2 = indices + bar_width / 2

        # Plot bars and curves for MPC/MPV and GMPC/GMPV with individual colors
        ax.bar(bar_positions_1, values1[j], bar_width, label='MPC' if i == 0 else 'MPV', color=colors1)
        ax.plot(bar_positions_1, values1[j], marker='o', color=list(cm.GnBu(np.linspace(0, 1, 32)))[18], linewidth=1.2)
        ax.bar(bar_positions_2, values2[j], bar_width, label='gMPC' if i == 0 else 'gMPV', color=colors2)
        ax.plot(bar_positions_2, values2[j], marker='o', color=list(cm.GnBu(np.linspace(0, 1, 32)))[26], linewidth=1.2)

        # Set the x-tick positions and labels
        ax.set_xticks(indices)
        ax.set_xticklabels(xticks)

        # Set titles for the first row of subplots
        if i == 0:
            ax.set_title(titles[j])

# Add legends
for ax in axes.flatten():
    ax.legend(loc='upper left', fontsize=16)
    # ax.legend(loc='upper left', frameon=True, facecolor='darkgray')


# Add vertical text labels to the left of the rows
fig.text(0.005, 0.75, '(a) Query Auto-completion', va='center', rotation='vertical', fontsize=20)
fig.text(0.005, 0.25, '(b) Movie Recommendation', va='center', rotation='vertical', fontsize=20)

# Adjust the layout and margins to prevent text and content from being cut off
fig.tight_layout(rect=[0.01, 0.01, 1, 1])

# Save and show the figure
plt.savefig('../fig/stochas.pdf')
plt.show()
plt.close()
