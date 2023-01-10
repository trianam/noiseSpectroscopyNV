import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

data = np.load("/tmp/plot.npz")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
pnt = ax.scatter(data['y0'], data['a'], data['w1'], c=data['c'], marker='o')
plt.colorbar(pnt)
ax.set_xlabel('y0')
ax.set_ylabel('a')
ax.set_zlabel('W1')
plt.show()
