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
X = numpy.array(X_)[:, 1:41]
#%%
import sklearn.decomposition


C = numpy.corrcoef(X.transpose())
plt.imshow(C, interpolation = 'NONE', cmap = plt.cm.viridis)
plt.gcf().set_size_inches((8, 8))
plt.savefig("/home/bbales2/cluster_expansion/images/correlation.png", dpi = 72)
plt.show()
#%%
#C = numpy.corrcoef(X.transpose())
#plt.imshow(C, interpolation = 'NONE', cmap = plt.cm.viridis)
seaborn.distplot(y, bins = 30)
plt.gcf().set_size_inches((8, 8))
plt.savefig("/home/bbales2/cluster_expansion/images/histogram.png", dpi = 72)
plt.show()
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
import json

with open('eci.json') as f:
    data = json.load(f)
#%%
ce = data['cluster_functions']
ecis = []
for c in ce:
    if 'eci' in c:
        ecis.append(c['eci'])
    else:
        ecis.append(0.0)
#%%

pca = sklearn.decomposition.PCA(40)
pca.fit(X)

X2 = pca.transform(X)

# Run the fit!
fit = m.vb(data = {
    "N" : X2.shape[0],
    "L" : X2.shape[1],
    "X" : X2[:, :],
    "y" : y
})#, algorithm = 'meanfield')

print fit
#%%
out = numpy.loadtxt('/tmp/tmpZ03QGO/output.csv', delimiter = ',', skiprows = 1)
#%%
def skip_comments(f):
    for line in f:
        if line[0] == '#'
            continue
        else:
            yield line
#%%
out = numpy.genfromtxt('/tmp/tmpZ03QGO/output.csv', delimiter = ',', comments = '#', names = True, skip_header = 15)
#%%

with open('/tmp/tmpZ03QGO/output.csv') as f:
    found_header = False
    header_length = 0
    for i, line in enumerate(f):
        line = line.strip()

        if found_header == False and line[0] != '#':
            labels = line.split(',')
            found_header = True
        elif found_header == True and line[0] != '#':
            header_length = i
            break

samples = numpy.loadtxt('/tmp/tmpZ03QGO/output.csv', skiprows = header_length, delimiter = ',')

import collections

counter = collections.Counter()

for i, label in enumerate(labels):
    tokens = label.split('.')

    if len(tokens) == 1:
        counter[label] += 1
    else:
        tlabel = "".join(label.split('.')[:-1])
        counter[tlabel] += 1

output = {}

for tlabel, count in counter.iteritems():
    idxs = numpy.where([label.startswith(tlabel) for label in labels])
    if count == 1:
        output[tlabel] = samples[:, idxs].reshape(-1)
    else:
        output[tlabel] = samples[:, idxs].reshape(-1, count)

#%%
    found_header = False

    samples = []
    for line in f:
        if line[0] == '#':
            continue

        if found_header == False:
            labels = line.strip().split(',')
            found_header = True

        try:
            samples.append([float(v) for v in line.strip().split(',')])
        except Exception as e:
            print e
            print line

samples = numpy.array(samples)

#%%
R = 24
means = samples[:, numpy.where([l[0] == 'w' for l in labels])].dot(pca.components_).mean(axis = 0)[:R].flatten()
stds = samples[:, numpy.where([l[0] == 'w' for l in labels])].dot(pca.components_).std(axis = 0)[:R].flatten()
mb = samples[:, numpy.where([l[0] == 'b' for l in labels])].mean()
sb = samples[:, numpy.where([l[0] == 'b' for l in labels])].std()

means = numpy.concatenate(([mb], means))
stds = numpy.concatenate(([sb], stds))

plt.plot(means, 'b*')
plt.plot(ecis[:R], 'go')
plt.plot(means - 2 * stds, 'r--')
plt.plot(means + 2 * stds, 'r--')
plt.gcf().set_size_inches((12, 12))
plt.plot()
#%%
means = fit.extract()['w'].dot(pca.components_).mean(axis = 0)[:R]
stds = fit.extract()['w'].dot(pca.components_).std(axis = 0)[:R]
mb = fit.extract()['b'].mean()
sb = fit.extract()['b'].std()

means = numpy.concatenate(([mb], means))
stds = numpy.concatenate(([sb], stds))

plt.plot(means, 'b*')
plt.plot(ecis[:R], 'go')
plt.plot(means - 2 * stds, 'r--')
plt.plot(means + 2 * stds, 'r--')
plt.gcf().set_size_inches((12, 12))
plt.plot()
#%%
X_ = numpy.array(X_)

results = {}

for f in range(10, X_.shape[1] - 1):
    X = X_[:, 1:f + 1]
    for d in range(10, f + 1):
        pca = sklearn.decomposition.PCA(d)
        pca.fit(X)

        X2 = pca.transform(X)

        lr = sklearn.linear_model.LinearRegression()

        lr.fit(X2, y)

        ecs = numpy.concatenate(([lr.intercept_], lr.coef_.dot(pca.components_)))

        results[(f, d)] = (ecs, numpy.std(lr.predict(X2) - y), lr.score(X2, y))

        #print results[(f, d)]

    print f
#%%
plt.plot(results[(127, 109)][0], 'r')
plt.plot(ecis, 'g--')

#%%
import scipy.spatial
ch = scipy.spatial.ConvexHull(zip(X[:, 0], y))

Xch, ych = X[ch.vertices, 0], y[ch.vertices]

midxs = numpy.argsort(Xch)

mXch = Xch[midxs]
mych = ych[midxs]
mvertices = ch.vertices[midxs]

yp = X.dot(means[1:]) + means[0]
ch = scipy.spatial.ConvexHull(zip(X[:, 0], yp))

Xch, ych = X[ch.vertices, 0], yp[ch.vertices]

#plt.plot(X[:, 0], yt, '*r')
#plt.plot(X[:, 0], y, '*b')
#plt.plot(Xch, ych)

idxs = numpy.argsort(Xch)

Xch = Xch[idxs]
ych = ych[idxs]
vertices = ch.vertices[idxs]

pych = X[mvertices, :].dot(means[1:]) + means[0]
errors = []

loss = 0.0
dlossdecis = numpy.zeros(means.shape[0] - 1)
for i, (xt, yt) in enumerate(zip(X[:, 0], yp)):
    j = bisect.bisect_left(mXch, xt)

    if j == 0:
        diff = yt - pych[j]
    else:
        dx = mXch[j] - mXch[j - 1]
        alpha = (xt - mXch[j - 1]) / dx
        diff = yt - (pych[j - 1] * (1 - alpha) + pych[j] * alpha)

    if diff < 0.0:
        loss += diff
        print i, dlossdecis
        dlossdecis += X[i, :]

plt.plot(mXch, mych, 'g')

others = numpy.random.randint(0, len(X), size = 5)

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
    pych = X[mvertices, :].dot(ecis[1:]) + means[0]

    yp = X.dot(ecis[1:]) + ecis[0]
    errors = []

    loss = 0.0
    dlossdecis = numpy.zeros(means.shape[0] - 1)
    for i, (xt, yt) in enumerate(zip(X[:, 0], yp)):
        j = bisect.bisect_left(mXch, xt)

        if j == 0:
            diff = yt - pych[j]
        else:
            dx = mXch[j] - mXch[j - 1]
            alpha = (xt - mXch[j - 1]) / dx
            diff = yt - (pych[j - 1] * (1 - alpha) + pych[j] * alpha)

        if diff < 0.0:
            loss += diff
            dlossdecis += X[i, :]

    return loss, dlossdecis

plt.plot(mXch, mych, 'g')

others = numpy.random.randint(0, len(X), size = 5)
#%%
plt.plot(X[ch.vertices, 0], y[ch.vertices])
plt.scatter(X[:, 0], y)
plt.show()
#%%
works = 0.1 * numpy.ones((189 - 10, 189 - 10))

for (f, d), (ecs, std, score) in results.iteritems():
    errors = []
    for i in range(len(ecs)):
        errors.append(ecis[i] - ecs[i])

    if numpy.any(numpy.abs(ecs) > 0.5):
        works[f - 10, d - 10] = 0.1
    else:
        works[f - 10, d - 10] = std#numpy.std(errors)#score

idx = numpy.unravel_index(numpy.argmin(works), works.shape)
print numpy.array(idx) + 10, works[idx]

plt.imshow(numpy.log(works), interpolation = 'NONE', cmap = plt.cm.viridis)
plt.colorbar()
#        1/0
#    y = numpy.array(y_)[:]
#    X = numpy.array()
#%%
ar = sklearn.linear_model.ARDRegression()

ar.fit(X_, y)
#%%
import sklearn.linear_model

lr = sklearn.linear_model.LinearRegression()
lr.fit(X[:, :], y)
plt.plot(lr.coef_, 'b')
plt.show()
print numpy.std(lr.predict(X) - y)
#%%
plt.plot(lr.coef_)
plt.ylabel('eV', fontsize = 16)
plt.savefig('/home/bbales2/cluster_expansion/images/least_squares.full.png')
plt.show()
print
#%%
#w = pca.components_.transpose().dot(fit.extract()['w'].transpose()).transpose()
w = fit.extract()['w'].dot(pca.components_)
#w = pca.inverse_transform(fit.extract()['w'])
means = w.mean(axis = 0)
stds = w.std(axis = 0)
plt.plot(means, 'r');
#plt.plot(lr.coef_, 'b')
plt.plot(means + 2 * stds, '--b')
plt.plot(means - 2 * stds, '--b')
#plt.legend(['Stan mean', 'LR'])
#plt.gcf().set_size_inches((12, 12))
plt.savefig('/home/bbales2/cluster_expansion/images/least_squares.full.png')
plt.show()

#w = fit.extract()['w']
C = numpy.corrcoef(w.transpose())
plt.imshow(C, cmap = plt.cm.viridis, interpolation = 'NONE')
plt.colorbar()
plt.show()
#%%
w = fit2.extract()['w']
means = w.mean(axis = 0)
stds = w.std(axis = 0)
plt.plot(means, 'r');
#plt.plot(lr.coef_, 'b')
plt.plot(means + 2 * stds, '--b')
plt.plot(means - 2 * stds, '--b')
#plt.legend(['Stan mean', 'LR'])
#plt.gcf().set_size_inches((12, 12))
plt.savefig('/home/bbales2/cluster_expansion/images/bayes_full.png')
plt.show()

#w = fit.extract()['w']
C = numpy.corrcoef(w.transpose())
plt.imshow(C, cmap = plt.cm.viridis, interpolation = 'NONE')
plt.colorbar()
plt.gcf().set_size_inches((8, 8))
plt.savefig('/home/bbales2/cluster_expansion/images/bayes_full_corr.png', dpi = 72)
plt.show()
#%%
n1 = 40
n2 = 20

pca = sklearn.decomposition.PCA(n1)
pca.fit(X)

X2 = pca.transform(X)

# Run the fit!
fit = m.sampling(data = {
    "N" : X.shape[0],
    "L" : X.shape[1],
    "X" : X[:, :],
    "y" : y
})

w1 = pca.components_[:].transpose().dot(fit.extract()['w'][:, :].transpose()).transpose()


pca = sklearn.decomposition.PCA(n2)
pca.fit(X)

X2 = pca.transform(X)

# Run the fit!
fit = m.sampling(data = {
    "N" : X.shape[0],
    "L" : X.shape[1],
    "X" : X[:, :],
    "y" : y
})
w2 = pca.components_[:].transpose().dot(fit.extract()['w'][:, :].transpose()).transpose()

#%%
n1 = 40
n2 = 20

pca = sklearn.decomposition.PCA(n1)
pca.fit(X)

X2 = pca.transform(X)

# Run the fit!
fit1 = m.sampling(data = {
    "N" : X2.shape[0],
    "L" : X2.shape[1],
    "X" : X2[:, :],
    "y" : y
})

#w1 = pca.components_[:].transpose().dot(fit1.extract()['w'][:, :].transpose()).transpose()
w1 = fit1.extract()['w']

pca = sklearn.decomposition.PCA(n2)
pca.fit(X)

X2 = pca.transform(X)

# Run the fit!
fit2 = m.sampling(data = {
    "N" : X2.shape[0],
    "L" : X2.shape[1],
    "X" : X2[:, :],
    "y" : y
})

#w2 = pca.components_[:].transpose().dot(fit2.extract()['w'][:, :].transpose()).transpose()
w2 = fit2.extract()['w']
#%%
means = w1.mean(axis = 0)
stds = w1.std(axis = 0)
plt.plot(means, 'r');
plt.plot(means + 2 * stds, 'g')
plt.plot(means - 2 * stds, 'g')

means = w2.mean(axis = 0)
stds = w2.std(axis = 0)
plt.plot(means, 'b');
plt.plot(means + 2 * stds, 'k')
plt.plot(means - 2 * stds, 'k')
plt.legend(['n = {0}'.format(n1), 'n = {0}'.format(n2)])
plt.gcf().set_size_inches((16, 16))
plt.show()
#%%

data = { 'feature' : [], 'pca_n' : [], 'value' : [] }
for i in range(40):#w1.shape[1]):
    for j in range(w1.shape[0]):
        data['feature'].append('w{0}'.format(i))
        data['pca_n'].append(n1)
        data['value'].append(w1[j, i])
        data['feature'].append('w{0}'.format(i))
        data['pca_n'].append(n2)
        data['value'].append(w2[j, i])

df = pandas.DataFrame(data)
#%%
seaborn.boxplot(x = "feature", y = "value", hue = "pca_n", data = df, palette = "Set3", linewidth = 1, fliersize = 0)
seaborn.despine(offset=10, trim=True)
plt.gcf().set_size_inches((16, 16))
#%%
# Plot the posterior predictive samples

yhat = fit.extract()['yhat']

means = yhat[:, :100].mean(axis = 0)
stds = yhat[:, :100].std(axis = 0)

plt.plot(means, 'rx')
plt.plot(means + 2 * stds, '-b')
plt.plot(means - 2 * stds, '-b')
plt.plot(y[:100], '*')
plt.gcf().set_size_inches((12, 12))
plt.show()

#%%
import seaborn
import pandas

# Plot the parameter posteriors!

a = fit.extract()['a']
b = fit.extract()['b']
sigma = fit.extract()['sigma']

plt.hist(a, 20, alpha = 0.5)
plt.hist(b, 20, alpha = 0.5)
plt.hist(sigma, 20, alpha = 0.5)
plt.legend(['a', 'b', 'sigma'])
plt.show()

df = pandas.DataFrame({ 'a' : a, 'b' : b, 'sigma' : sigma })

seaborn.pairplot(df)
plt.show()
#%%
#C = numpy.corrcoef(y.transpose())
w = fit.extract()['w']
C = numpy.corrcoef(w.transpose())
plt.imshow(C, interpolation = 'NONE')
plt.show()

#%%
import sklearn.decomposition

pca = sklearn.decomposition.PCA()
pca.fit(X)

#%%
X2 = pca.transform(X)

C = numpy.corrcoef(X2.transpose())
plt.imshow(C, interpolation = 'NONE')
plt.show()
#%%
seaborn.distplot(y, bins = 30, norm_hist = True, kde = False)
seaborn.distplot(fit2.extract()['yhat'][-500:].flatten(), bins = 30, kde = False, norm_hist = True)
plt.gcf().set_size_inches((8, 8))
plt.legend(['Posterior predictive', 'y'], fontsize = 20)
plt.savefig("/home/bbales2/cluster_expansion/images/posterior_predictive_full.png", dpi = 72)
plt.show()