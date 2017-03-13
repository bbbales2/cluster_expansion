#%%

import numpy
import scipy.stats
import matplotlib.pyplot as plt
import os
import sklearn.decomposition
import seaborn
import pandas
import json

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
X = numpy.array(X_)[:, 1:41]


with open('eci.json') as f:
    data = json.load(f)

ce = data['cluster_functions']
ecis = []
for c in ce:
    if 'eci' in c:
        ecis.append(c['eci'])
    else:
        ecis.append(0.0)

#%%

# This is our very high quality Stan model

import pystan

model = """
data {
    int<lower=0> N;
    int<lower=0> L;
    matrix[N, L] X;
    vector[N] y;
}

parameters {
    vector[L] w;
    real b;
    real<lower = 0.0> sigma;
}

model {
    y ~ normal(X * w + b, sigma);
}

generated quantities {
    vector[N] yhat;

    for (i in 1:N)
        yhat[i] = normal_rng(X[i] * w + b, sigma);
}
"""

m = pystan.StanModel(model_code = model)

#%%

# Run the fit!
fit2 = m.sampling(data = {
    "N" : X.shape[0],
    "L" : X.shape[1],
    "X" : X[:, :],
    "y" : y
})

print fit2
#%%
R = 40
means = fit2.extract()['w'].mean(axis = 0)[:R]
stds = fit2.extract()['w'].std(axis = 0)[:R]
mb = fit2.extract()['b'].mean()
sb = fit2.extract()['b'].std()

means = numpy.concatenate(([mb], means))
stds = numpy.concatenate(([sb], stds))

plt.plot(ecis[:R], 'g*')
plt.plot(means, 'b*')
plt.plot(means - 2 * stds, 'r--')
plt.plot(means + 2 * stds, 'r--')
plt.gcf().set_size_inches((12, 12))
plt.plot()
#%%
import scipy.spatial
import bisect

ch = scipy.spatial.ConvexHull(zip(X[:, 0], y))

Xch, ych = X[ch.vertices, 0], y[ch.vertices]

midxs = numpy.argsort(Xch)

mXch = Xch[midxs]
mych = ych[midxs]
mvertices = ch.vertices[midxs]

def UgradU(ecis):
    pych = X[mvertices, :].dot(ecis[1:]) + ecis[0]

    yp = X.dot(ecis[1:]) + ecis[0]

    loss = 0.0
    dlossdecis = numpy.zeros(means.shape[0])
    for i, (xt, yt) in enumerate(zip(X[:, 0], yp)):
        j = bisect.bisect_left(mXch, xt)

        if j == 0:
            diff = yt - pych[j]

            if diff < 0.0:
                loss += diff
                dlossdecis[1:] += X[i, :] - X[mvertices[j], :]
        else:
            dx = mXch[j] - mXch[j - 1]
            alpha = (xt - mXch[j - 1]) / dx
            diff = yt - (pych[j - 1] * (1 - alpha) + pych[j] * alpha)

            if diff < 0.0:
                loss += diff
                dlossdecis[1:] += X[i, :] - (1 - alpha) * X[mvertices[j - 1], :] - alpha * X[mvertices[j], :]

    return loss, dlossdecis

l, dl = UgradU(means)

#%%
fddl = numpy.zeros(len(means))
for i in range(len(means)):
    l1, _ = UgradU(means)

    d = 0.0001
    dmeans = means.copy()
    dmeans[i] += d

    l2, _ = UgradU(dmeans)

    fddl[i] = (l2 - l1) / d

print "Derivative, FD check"
for v1, v2 in zip(dl, fddl):
    print "{0:.3f} {0:.3f}".format(v1, v2)
