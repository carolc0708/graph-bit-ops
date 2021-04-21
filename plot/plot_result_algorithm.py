import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotinpy as pnp

# labels = ['h09', 'i04', 'coAuth', 'coPaper', 'citPat', 'Journals', 'G43', 'ship_003']
labels = ['bel_osm', 'del_24', 'road_cent', 'road_CA', 'road_usa']

# BFS
# pascal = [0.13,	1.81, 0.92,	0.70, 4.44, 4.08, 3.18,	7.22]
# volta = [0.23, 2.02, 1.92, 1.08, 9.05, 4.80, 3.41, 5.39]
# pascal = [1.00,	0.20,	0.16,	0.96,	0.16]
# volta = [1.37,	0.36,	0.28,	1.54,	0.26]

# # SSSP
# pascal = [0.40,	0.92,	1.07,	0.86,	0.00,	13.11,	5.87,	8.24]
# volta = [0.52,	1.77,	1.35,	1.61,	0.00,	16.91,	8.07,	15.52]
# pascal = [0.80,	1.55,	0.45,	0.74,	0.85]
# volta = [1.45,	2.63,	0.61,	1.26,	1.12]

# # PR
# scale_free = [0.11, 0.68]
# mesh_like = [0.66, 0.09]
#
# # TC
# pascal = [0.31,	1708.01,	0.30,	0.42,	0.13,	40.50,	0.23,	6.41]
# volta = [0.57,	1833.30,	0.26,	0.49,	0.29,	26.90,	0.29,	4.82]
pascal = [1.20,	0.68,	0.48,	0.69,	0.55]
volta = [0.47,	0.51,	0.19,	0.42,	0.33]

# volta-baseline
# speedup_4 = [11.48,	0.56,	8.39,	1.00,	51.12,	1.02,	0.53,	5.07]
# speedup_8 = [4.55,	0.59,	3.38,	0.51,	11.53,	1.26,	0.14,	5.48]
# speedup_16 = [14.89, 4.72,	5.94,	1.67,	16.44,	9.40,	0.90,	30.73]
# speedup_32 = [5.76,	4.44,	1.19,	0.63,	2.66,	15.67,	1.42,	16.96]
# speedup_64 = [1.03,	1.00,	0.10,	0.09,	0.18,	10.07,	0.83,	3.70]


# len(labels)
x = np.arange(len(labels)) # the label locations
# print(x)
width = 0.15  # the width of the bars

fig, ax = plt.subplots(dpi=300)
rects1 = ax.bar(x-width/2, pascal, width, label='Pascal', color='#829cbc')
rects2 = ax.bar(x+width/2, volta, width, label='Volta', color='#376996')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Speedup over GraphBLAST', fontsize=16)
# ax.set_title('Speedup by block size over cuSPARSE')
ax.set_xticks(x)
ax.legend(prop={"size":16})
ax.tick_params(labelsize=16)
ax.set_xticklabels(labels, fontsize=12) #12

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
#                     ha='center', va='bottom', fontsize=16) #10
#
#
# autolabel(rects1)
# autolabel(rects2)

fig.tight_layout()

# plt.savefig("bfs_scale_free.jpeg")
# plt.savefig("sssp_scale_free.jpeg")
# plt.savefig("tc_scale_free.jpeg")
# plt.savefig("bfs_mesh_like.jpeg")
# plt.savefig("sssp_mesh_like.jpeg")
plt.savefig("tc_mesh_like.jpeg")

#plt.show()