#%%

import numpy
import scipy.stats
import matplotlib.pyplot as plt
import os
import sklearn.decomposition
import seaborn
import pandas

os.chdir('/home/bbales2/cluster_expansion')

y = {}
with open('y.csv') as f:
    f.readline()

    for line in f:
        line = line.strip()

        if len(line) == 0:
            continue

        data = line.split(',')

        name = data[0]
        e = float(data[1])

        y[name] = e

X = {}
with open('X.csv') as f:
    f.readline()

    for line in f:
        line = line.strip()

        if len(line) == 0:
            continue

        data = line.split(',')

        name = data[0]
        e = numpy.array([float(d) for d in data[1:]])

        X[name] = e

print set(y.keys()) - set(X.keys())

skeys = [key for key, v in sorted(y.iteritems(), key = lambda x : x[1])]
#skeys = sorted(y.keys())

y_ = []
X_ = []

for key in skeys:
    y_.append(y[key])
    X_.append(X[key])

y = numpy.array(y_)[:]
X = numpy.array(X_)[:, 1:]
#%%
ylow = y[y < -0.23]
Xlow = X[y < -0.23]
yhigh = y[y >= -0.23]
Xhigh = X[y >= -0.23]


C = numpy.corrcoef(Xlow.transpose())
plt.imshow(C, interpolation = 'NONE', cmap = plt.cm.viridis)
plt.gcf().set_size_inches((8, 8))
#plt.savefig("/home/bbales2/cluster_expansion/images/low_correlation.png", dpi = 72)
plt.show()

C = numpy.corrcoef(Xhigh.transpose())
plt.imshow(C, interpolation = 'NONE', cmap = plt.cm.viridis)
plt.gcf().set_size_inches((8, 8))
#plt.savefig("/home/bbales2/cluster_expansion/images/high_correlation.png", dpi = 72)
plt.show()
