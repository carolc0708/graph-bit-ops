import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# mesh-like
labels = ['bel_osm', 'del_24', 'road_cent', 'road_CA', 'road_usa']
# Pascal-baseline
# speedup_8 = [1.18, 1.04, 0.84, 0.88, 0.64]
# speedup_16 = [0.88, 1.00, 0.56, 0.74, 0.54]
# speedup_32 = [0.83, 1.12, 0.49, 0.73, 0.54]
# speedup_64 = [0.41, 0.79, 0.24, 0.41, 0.33]

# Volta-baseline
# speedup_8 = [1.36, 1.29, 1.44, 1.28, 0.98]
# speedup_16 = [1.15, 1.09, 0.85, 1.04, 0.77]
# speedup_32 = [0.94, 0.95, 0.56, 0.81, 0.59]
# speedup_64 = [0.42, 0.60, 0.23, 0.39, 0.32]

# Turing-baseline
# speedup_8 = [1.26, 0.95, 1.16, 1.09, 0.75]
# speedup_16 = [1.04, 0.85, 0.79, 0.97, 0.62]
# speedup_32 = [1.01, 0.83, 0.64, 0.87, 0.55]
# speedup_64 = [0.44, 0.58, 0.26, 0.42, 0.32]

# Pascal-new
speedup_8 = [1.14, 1.10, 0.92, 0.88, 0.76]
speedup_16 = [0.87, 1.05, 0.61, 0.76, 0.63]
speedup_32 = [0.80, 1.13, 0.50, 0.72, 0.60]
speedup_64 = [0.40, 0.75, 0.25, 0.41, 0.34]

x = np.arange(len(labels))  # the label locations
width = 0.20  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x-width, speedup_8, width, label='8x8', color='darkseagreen')
rects2 = ax.bar(x, speedup_16, width, label='16x16', color='cadetblue')
rects3 = ax.bar(x+width, speedup_32, width, label='32x32', color='steelblue')
rects4 = ax.bar(x+width*2, speedup_64, width, label='64x64', color='slategray')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Speedup over cuSPARSE-csrmv-float')
# ax.set_title('Speedup by block size over cuSPARSE')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

## Add line at 1.0 speedup
plt.axhline(y=1, color='gray', linestyle='--')


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=5)


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()

plt.savefig("bmv_mesh_like.png")
#plt.show()