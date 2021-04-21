import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# mesh-like
labels = ['bel_osm', 'del_24', 'road_cent', 'road_CA', 'road_usa']

# Pascal-baseline
speedup_4 = [6.51,	5.11,	10.62,	7.52,	0.76]
speedup_8 = [10.10,	5.07,	7.82,	8.39,	0.74]
speedup_16 = [4.78,	2.47,	2.18,	4.19,	0.32]
speedup_32 = [2.13,	1.14,	0.60,	1.67,	0.13]
speedup_64 = [0.30,	0.24,	0.09,	0.34,	0.03]

# Volta-baseline
# speedup_4 = [0.29,	0.47,	0.85,	0.43,	0.47]
# speedup_8 = [0.65,	0.69,	0.97,	0.84,	0.87]
# speedup_16 = [1.24,	1.62,	1.83,	1.83,	1.59]
# speedup_32 = [1.76,	0.93,	0.47,	1.36,	0.91]
# speedup_64 = [0.17,	0.14,	0.05,	0.19,	0.12]

# Turing-baseline
# speedup_4 = [0.52,	0.84,	1.32,	0.76,	0.80]
# speedup_8 = [1.01,	0.73,	1.04,	1.33,	1.02]
# speedup_16 = [2.13,	1.65,	1.52,	2.94,	2.00]
# speedup_32 = [2.57,	1.14,	0.59,	2.24,	1.18]
# speedup_64 = [0.32,	0.22,	0.08,	0.35,	0.19]

x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots(figsize=(8,6), dpi=300)

rects1 = ax.bar(x-width*2, speedup_4, width, label='4x4', color='#ea8c55')
rects2 = ax.bar(x-width, speedup_8, width, label='8x8', color='#c75146')
rects3 = ax.bar(x, speedup_16, width, label='16x16', color='#ad2e24')
rects4 = ax.bar(x+width, speedup_32, width, label='32x32', color='#81171b')
rects5 = ax.bar(x+width*2, speedup_64, width, label='64x64', color='#540804')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Speedup over cuSPARSE-csrSpGEMM-float', fontsize=16)
# ax.set_title('Speedup by block size over cuSPARSE')
ax.set_xticks(x)
# ax.set_ylim(0, 11)
ax.tick_params(labelsize=16)
ax.legend(prop={"size":16}, loc="upper right")#, bbox_to_anchor=(0.9,0.9))
ax.set_xticklabels(labels, fontsize=16)

## Add line at 1.0 speedup
plt.axhline(y=1, color='gray', linestyle='--')


# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom', fontsize=16)
#
#
# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)
# autolabel(rects4)
# autolabel(rects5)

fig.tight_layout()

plt.savefig("pascal_bmm_mesh_like.jpeg")
#plt.show()