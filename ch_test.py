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
X = numpy.array(X_)[:, 1:]


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

#mXch = Xch[midxs]
#mych = ych[midxs]
mvertices = ch.vertices[midxs]

def UgradU(ecis):
    pych = X[mvertices, :len(ecis) - 1].dot(ecis[1:]) + ecis[0]
    mXch = X[mvertices, 0]

    yp = X[:, :len(ecis) - 1].dot(ecis[1:]) + ecis[0]

    #plt.plot(mXch, pych)
    #plt.plot(X[:, 0], yp, 'r*')
    #plt.show()

    loss = 0.0
    dlossdecis = numpy.zeros(len(ecis))
    for i, (xt, yt) in enumerate(zip(X[:, 0], yp)):
        j = bisect.bisect_left(mXch, xt)

        if j == 0:
            diff = yt - pych[j]

            if diff < 0.0:
                loss += diff
                dlossdecis[1:] += X[i, :len(ecis) - 1] - X[mvertices[j], :len(ecis) - 1]
        else:
            dx = mXch[j] - mXch[j - 1]
            alpha = (xt - mXch[j - 1]) / dx
            diff = yt - (pych[j - 1] * (1 - alpha) + pych[j] * alpha)

            if diff < 0.0:
                loss += diff
                dlossdecis[1:] += X[i, :len(ecis) - 1] - (1 - alpha) * X[mvertices[j - 1], :len(ecis) - 1] - alpha * X[mvertices[j], :len(ecis) - 1]

    return loss, dlossdecis

l, dl = UgradU(means)

#%%
X = numpy.array(X)

results = {}

for f in range(10, X.shape[1] - 1):
    Xt = X[:, 0:f + 1]
    for d in range(10, f + 1):
        pca = sklearn.decomposition.PCA(d)
        pca.fit(Xt)

        X2 = pca.transform(Xt)

        lr = sklearn.linear_model.LinearRegression()

        lr.fit(X2, y)

        ecs = numpy.concatenate(([lr.intercept_], lr.coef_.dot(pca.components_)))

        results[(f, d)] = (ecs, numpy.std(lr.predict(X2) - y), lr.score(X2, y), UgradU(ecs)[0])

    print f

#%%
works = 0.1 * numpy.ones((189 - 10, 189 - 10))

for (f, d), (ecs, std, score, err) in results.iteritems():
    errors = []
    for i in range(len(ecs)):
        errors.append(ecis[i] - ecs[i])

    if numpy.any(numpy.abs(ecs) > 0.5):
        works[f - 10, d - 10] = -1.0
    else:
        works[f - 10, d - 10] = err#numpy.std(errors)#score

idx = numpy.unravel_index(numpy.argmin(works), works.shape)
print numpy.array(idx) + 10, works[idx]

plt.imshow(works, interpolation = 'NONE', cmap = plt.cm.viridis)
plt.colorbar()
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
