import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def measure(n):

    "Measurement model, return two coupled measurements."

    m1 = np.random.normal(size=n)

    m2 = np.random.normal(scale=0.5, size=n)

    return m1+m2, m1-m2

m1, m2 = measure(2000)

xmin = m1.min()

xmax = m1.max()

ymin = m2.min()

ymax = m2.max()


X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

positions = np.vstack([X.ravel(), Y.ravel()])

values = np.vstack([m1, m2])

kernel = stats.gaussian_kde(values, bw_method=0.1)

Z = np.reshape(kernel(positions).T, X.shape)



fig, ax = plt.subplots()

plt.pcolormesh(X, Y, Z)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
cmap = matplotlib.cm.get_cmap(None)
ax.set_facecolor(cmap(0.))
ax.invert_yaxis()
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax.set_title("test")
plt.show()
