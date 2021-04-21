import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotinpy as pnp

# scale-free
labels = ['h09', 'i04', 'coAuth', 'coPaper', 'citPat', 'Journals', 'G43', 'ship_003']
# pascal-baseline
# speedup_4 = [40.76,	1.08,	48.80,	49.52,	267.66,	5.89,	6.36,	28.71]
# speedup_8 = [23.52,	2.24,	27.74,	36.86,	93.98,	6.59,	2.06,	37.49]
# speedup_16 = [7.87,	2.60,	7.32,	12.77,	14.67,	7.47,	1.18,	24.10]
# speedup_32 = [3.15,	2.49,	1.34,	4.86,	2.71,	7.47,	0.99,	11.99]
# speedup_64 = [0.87,	0.85,	0.18,	1.02,	0.00,	6.22,	1.03,	4.18]

# volta-baseline
speedup_4 = [11.48,	0.56,	8.39,	1.00,	51.12,	1.02,	0.53,	5.07]
speedup_8 = [4.55,	0.59,	3.38,	0.51,	11.53,	1.26,	0.14,	5.48]
speedup_16 = [14.89, 4.72,	5.94,	1.67,	16.44,	9.40,	0.90,	30.73]
speedup_32 = [5.76,	4.44,	1.19,	0.63,	2.66,	15.67,	1.42,	16.96]
speedup_64 = [1.03,	1.00,	0.10,	0.09,	0.18,	10.07,	0.83,	3.70]

# Turing-baseline
# speedup_4 = [10.99,	0.50,	9.33,	6.85,	67.75,	0.92,	0.52,	6.00]
# speedup_8 = [4.72,	0.58,	4.66,	4.36,	16.98,	1.08,	0.13,	7.10]
# speedup_16 = [13.65, 4.18,	9.71,	13.90,	20.11,	7.38,	0.80,	42.49]
# speedup_32 = [6.24,	4.84,	2.40,	6.29,	4.38,	12.00,	1.24,	26.43]
# speedup_64 = [1.15,	1.14,	0.22,	0.96,	0.00,	7.38,	0.66,	6.21]


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
#                     ha='center', va='bottom', fontsize=16)
#
#
# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)
# autolabel(rects4)
# autolabel(rects5)

fig.tight_layout()

plt.savefig("volta_bmm_scale_free.jpeg")
#plt.show()