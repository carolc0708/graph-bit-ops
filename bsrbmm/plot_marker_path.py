import matplotlib
import matplotlib.pyplot as plt
import numpy as np



fig, ax = plt.subplots()
ax.set_ylabel('Non-empty Tiles to the Total Tiles(%)', fontsize=14)
ax.set_xlabel('Tile Dimension', fontsize=14)
xpoints = np.array([4, 8, 16, 32, 64])
ypoints = np.array([0.05, 0.15, 0.40, 1.10, 3.10])
plt.plot(xpoints, ypoints, "D--", color='#4281a4', markersize=8)
plt.savefig("tile_increment_path.png")
#plt.show()