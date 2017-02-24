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
import sklearn.decomposition


C = numpy.corrcoef(X.transpose())
plt.imshow(C, interpolation = 'NONE', cmap = plt.cm.viridis)
plt.gcf().set_size_inches((8, 8))
plt.savefig("/home/bbales2/cluster_expansion/images/correlation.png", dpi = 72)
plt.show()
#%%
#C = numpy.corrcoef(X.transpose())
#plt.imshow(C, interpolation = 'NONE', cmap = plt.cm.viridis)
seaborn.distplot(y)
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
    w ~ double_exponential(0, 0.1);

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

pca = sklearn.decomposition.PCA()
pca.fit(X)

X2 = pca.transform(X)

# Run the fit!
fit = m.sampling(data = {
    "N" : X2.shape[0],
    "L" : X2.shape[1],
    "X" : X2[:, :],
    "y" : y
})

print fit
#%%
import sklearn.linear_model

lr = sklearn.linear_model.LinearRegression()
lr.fit(X[:, :], y)
plt.plot(lr.coef_, 'b')
plt.show()
print numpy.std(lr.predict(X) - y)

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
plt.savefig('/home/bbales2/cluster_expansion/images/bayes_prior_01.png', dpi = 72)
plt.show()

#w = fit.extract()['w']
C = numpy.corrcoef(w.transpose())
plt.imshow(C, cmap = plt.cm.viridis, interpolation = 'NONE')
plt.colorbar()
plt.gcf().set_size_inches((8, 8))
%plt.savefig('/home/bbales2/cluster_expansion/images/bayes_prior_01.0_corr.png', dpi = 72)
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
plt.savefig('/home/bbales2/cluster_expansion/images/bayes_prior_01.png', dpi = 72)
plt.show()

C = numpy.corrcoef(w.transpose())
plt.imshow(C, cmap = plt.cm.viridis, interpolation = 'NONE')
plt.colorbar()
plt.gcf().set_size_inches((8, 8))
plt.savefig('/home/bbales2/cluster_expansion/images/bayes_prior_01_corr.png', dpi = 72)
plt.show()

#%%

seaborn.distplot(y, bins = 30, norm_hist = True, kde = False)
seaborn.distplot(fit2.extract()['yhat'][-500:].flatten(), bins = 30, kde = False, norm_hist = True)
plt.gcf().set_size_inches((8, 8))
plt.legend(['Posterior predictive', 'y'], fontsize = 20)
#plt.savefig("/home/bbales2/cluster_expansion/images/posterior_predictive_full.png", dpi = 72)
plt.show()
