import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set seaborn theme and matplotlib parameters
sns.set_theme(style="whitegrid")
plt.rcParams['pdf.fonttype'] = 42

# Define size parameters
legend_size = 28
marker_size = 14
line_width = 3
axis_label_size = 30
value_size = 32

# Define x and y labels
x_labels = ["DA-SS", "GA-SS", r'GA-SS$_{\sum \prod}$', r'GA-SS$_{\prod \sum}$']
y_labels = ["DA-SS", "GA-SS", r'GA-SS$_{\sum \prod}$', r'GA-SS$_{\prod \sum}$']

# Generate data with increasing values for each row and column
# data1 = np.array([np.array([i + j for j in np.linspace(0, 0.5, num=6)]) for i in np.linspace(0, 0.5, num=6)])
data1 = [
    np.array([1.000, 0.143, 0.103, 0.089]),
    np.array([0.143, 1.000, 0.543, 0.575]),
    np.array([0.103, 0.543, 1.000, 0.897]),
    np.array([0.089, 0.575, 0.897, 1.000]),
]
data1 = np.vstack(data1)
data2 = [
    np.array([1.000, 0.213, 0.172, 0.106]),
    np.array([0.213, 1.000, 0.632, 0.609]),
    np.array([0.172, 0.632, 1.000, 0.903]),
    np.array([0.106, 0.609, 0.903, 1.000]),
]
data2 = np.vstack(data2)

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(23, 12))

# Create a colormap
cmap = sns.cubehelix_palette(as_cmap=True)

# Generate heatmaps
ax1=sns.heatmap(data1, cmap=cmap, linewidths=0.1, linecolor='white', annot=True, fmt='.3f',
            annot_kws={'size': value_size}, cbar=True, ax=ax1, xticklabels=x_labels, yticklabels=y_labels,
            cbar_kws={'orientation': 'horizontal', 'pad': 0.08, 'location': 'top'})
ax2=sns.heatmap(data2, cmap=cmap, linewidths=0.1, linecolor='white', annot=True, fmt='.3f',
            annot_kws={'size': value_size}, cbar=True, ax=ax2, xticklabels=x_labels, yticklabels=y_labels,
            cbar_kws={'orientation': 'horizontal', 'pad': 0.08, 'location': 'top'})

# Adjust space between heatmaps
plt.subplots_adjust(wspace=0.3)

# use matplotlib.colorbar.Colorbar object
cbar1 = ax1.collections[0].colorbar
cbar1.ax.tick_params(labelsize=legend_size)
cbar2 = ax2.collections[0].colorbar
cbar2.ax.tick_params(labelsize=legend_size)


# # Increase legend size
# cbar_ax = fig.add_axes([0.2, 0.2, 0.02, 0.7])  # position of the colorbar (left, bottom, width, height)
# fig.colorbar(ax1.get_children()[0], cax=cbar_ax)
# cbar_ax.tick_params(labelsize=legend_size)


# Increase size of words on x-axis, y-axis
ax1.tick_params(axis='both', which='major', labelsize=axis_label_size)
ax2.tick_params(axis='both', which='major', labelsize=axis_label_size)

# Set titles
ax1.set_title('(a) Query Auto-completion', fontsize=54, y=-0.04, pad=-75)
ax2.set_title('(b) Movie Recommendation', fontsize=54, y=-0.04, pad=-75)


# rotate y labels
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=20)
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=20)

# rotate y=x labels
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)

fig.subplots_adjust(wspace=0.3)
fig.tight_layout()

plt.savefig('../fig/correlation.pdf')
plt.show()
plt.close()
plt.clf()
