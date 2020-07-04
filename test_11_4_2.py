import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

n = 200

x = np.linspace(-10, 10, n)
y = np.linspace(-10, 10, n)

print(x, y)

X, Y = np.meshgrid(x, y)
Z = X + Y

cm_bg = mpl.colors.ListedColormap(["#ffa0a0", "#a0ffa0", "#a0a0ff"])
plt.pcolormesh(X, Y, Z, cmap=cm_bg)     #X, Y, 决定网格的位置, Z决定网格的颜色
plt.show()