import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# scale-free
labels = ['h09', 'i04', 'coAuth', 'coPaper', 'citPat', 'Journals', 'G43', 'ship_003']
# pascal-baseline
# speedup_4 = [1.06, 3.23, 1.57, 2.22, 1.93, 1.57, 0.62, 2.14]
# speedup_8 = [1.00, 4.15, 1.48, 2.51, 1.54, 1.83, 0.53, 3.14]
# speedup_16 = [0.79, 4.18, 1.15, 2.12, 0.82, 1.83, 0.73, 3.44]
# speedup_32 = [0.76, 4.82, 0.99, 2.13, 0.56, 2.20, 1.00, 4.07]
# speedup_64 = [0.40, 2.97, 0.47, 1.03, 0.20, 2.20, 1.00, 3.01]

# volta-baseline
speedup_4 = [1.67, 3.56, 1.71, 2.25, 2.67, 1.33, 0.60, 3.16]
speedup_8 = [1.47, 4.76, 1.35, 2.71, 1.84, 1.60, 0.50, 6.13]
speedup_16 = [1.15, 4.06, 1.07, 2.30, 0.94, 2.00, 0.60, 4.26]
speedup_32 = [0.86, 3.42, 0.88, 1.75, 0.49, 2.00, 1.00, 3.77]
speedup_64 = [0.39, 1.77, 0.34, 0.70, 0.14, 1.60, 0.86, 2.33]

# Turing-baseline
# speedup_4 = [1.82, 3.18, 0.99, 2.58, 2.55, 1.20, 0.56, 3.11]
# # speedup_8 = [1.64, 4.12, 1.15, 2.92, 1.92, 1.50, 0.50, 5.50]
# # speedup_16 = [1.43, 4.31, 1.00, 2.60, 1.07, 2.00, 0.83, 4.93]
# # speedup_32 = [1.17, 4.23, 0.73, 2.16, 0.63, 2.00, 0.83, 4.47]
# # speedup_64 = [0.53, 2.45, 0.33, 0.88, 0.18, 0.86, 1.00, 2.86]

# pascal-new
# speedup_8 = [0.76, 4.31, 1.60, 2.33, 1.59, 1.83, 0.21, 2.56]
# speedup_16 = [0.60, 4.26, 1.26, 1.97, 0.85, 2.20, 0.33, 2.89]
# speedup_32 = [0.57, 4.69, 1.08, 2.03, 0.57, 2.20, 0.62, 3.49]
# speedup_64 = [0.33, 3.07, 0.46, 0.97, 0.20, 2.20, 0.80, 2.42]

x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots(figsize=(8,6), dpi=300)
rects1 = ax.bar(x-width*2, speedup_4, width, label='4x4', color='#cfe0c3')
rects2 = ax.bar(x-width, speedup_8, width, label='8x8', color='#9ec1a3')
rects3 = ax.bar(x, speedup_16, width, label='16x16', color='#70a9a1')
rects4 = ax.bar(x+width, speedup_32, width, label='32x32', color='#40798c')
rects5 = ax.bar(x+width*2, speedup_64, width, label='64x64', color='#1f363d')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Speedup over cuSPARSE-csrmv-float', fontsize=16)
# ax.set_title('Speedup by block size over cuSPARSE')
ax.set_xticks(x)
ax.tick_params(labelsize=16)
ax.legend(prop={"size":16}, loc="upper right")
ax.set_xticklabels(labels, fontsize=12)

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
#                     ha='center', va='bottom', fontsize=6)
#
#
# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)
# autolabel(rects4)
# autolabel(rects5)

fig.tight_layout()

plt.savefig("volta_bmv_scale_free.jpeg")
#plt.show()