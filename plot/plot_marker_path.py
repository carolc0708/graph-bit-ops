import matplotlib
import matplotlib.pyplot as plt
import numpy as np

y1 = np.array([0.06, 0.16, 0.43, 1.18, 3.15])
y2 = np.array([0.00, 0.00, 0.00, 0.01, 0.01])
y3 = np.array([0.02, 0.06, 0.21, 0.77, 2.89])
y4 = np.array([0.04, 0.09, 0.21, 0.58, 1.68])
y5 = np.array([0.00, 0.01, 0.03, 0.11, 0.44])
y6 = np.array([99.79, 100.00, 100.00, 100.00, 100.00])
y7 = np.array([27.64, 72.19, 98.99, 99.80, 100.00])
y8 = np.array([0.12, 0.19, 0.33, 0.64, 1.27])
y9 = np.array([0.00, 0.00, 0.01, 0.02, 0.08])
y10 = np.array([0.00, 0.00, 0.00, 0.00, 0.01])
y11 = np.array([0.00, 0.00, 0.00, 0.01, 0.03])
y12 = np.array([0.00, 0.00, 0.01, 0.03, 0.07])
y13 = np.array([0.00, 0.00, 0.00, 0.00, 0.01])


fig, ax = plt.subplots(figsize=(8,6), dpi=300)
ax.set_ylabel('Num of Non-empty Tiles to Total Tiles (%)', fontsize=16)
ax.set_xlabel('Tile Dimension (NxN)', fontsize=16)
xpoints = np.array([4, 8, 16, 32, 64])
ax.set_xticks(xpoints)
ax.tick_params(labelsize=16)

plt.plot(xpoints, y1, "D--", color='#f94144', markersize=5, label='h09')
plt.plot(xpoints, y2, "D--", color='#f3722c', markersize=5, label='i04')
plt.plot(xpoints, y3, "D--", color='#f8961e', markersize=5, label='coAuth')
plt.plot(xpoints, y4, "D--", color='#f9844a', markersize=5, label='coPaper')
plt.plot(xpoints, y5, "D--", color='#f9c74f', markersize=5, label='citPat')
plt.plot(xpoints, y6, "D--", color='#90be6d', markersize=5, label='Journals')
plt.plot(xpoints, y7, "D--", color='#43aa8b', markersize=5, label='G43')
plt.plot(xpoints, y8, "D--", color='#4d908e', markersize=5, label='ship_003')
plt.plot(xpoints, y9, "D--", color='#577590', markersize=5, label='bel_osm')
plt.plot(xpoints, y10, "D--", color='#277da1', markersize=5, label='del_24')
plt.plot(xpoints, y11, "D--", color='#f94144', markersize=5, label='road_cent')
plt.plot(xpoints, y12, "D--", color='#f3722c', markersize=5, label='road_CA')
plt.plot(xpoints, y13, "D--", color='#f8961e', markersize=5, label='road_usa')
ax.legend(prop={"size":14}, loc="best")

plt.savefig("tile_increment_path.jpeg")
#plt.show()