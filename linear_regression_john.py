#%%

import numpy
import scipy.stats
import matplotlib.pyplot as plt
import os
import sklearn.decomposition
import seaborn
import pandas
import pystan
import json

os.chdir('/home/bbales2/cluster_expansion')

with open('eci.json') as f:
    data = json.load(f)

ce = data['cluster_functions']
ecis = []
for c in ce:
    if 'eci' in c:
        ecis.append(c['eci'])
    else:
        ecis.append(0.0)

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
X = numpy.array(X_)[:, 1:21]
#%%

m = pystan.StanModel("models/linear_regression.stan")

#%%

# Run the fit!
fit = m.sampling(data = {
    "N" : X.shape[0],
    "L" : X.shape[1],
    "X" : X[:, :],
    "y" : y
})

print fit

# You could also run an optimizer on the problem if you want (to get a MAP estimate)
#fit2 = m.optimizing(data = {
#    "N" : X.shape[0],
#    "L" : X.shape[1],
#    "X" : X[:, :],
#    "y" : y
#})
#%%
R = X.shape[1]
w_median = numpy.median(fit.extract()['w'], axis = 0)
w_lower = numpy.percentile(fit.extract()['w'], 2.5, axis = 0)
w_upper = numpy.percentile(fit.extract()['w'], 97.5, axis = 0)

b_median = numpy.median(fit.extract()['b'], axis = 0)
b_lower = numpy.percentile(fit.extract()['b'], 2.5, axis = 0)
b_upper = numpy.percentile(fit.extract()['b'], 97.5, axis = 0)

median = numpy.concatenate(([b_median], w_median))
lower = numpy.concatenate(([b_lower], w_lower))
upper = numpy.concatenate(([b_upper], w_upper))

plt.plot(lower, 'r.')
plt.plot(upper, 'r.')
plt.plot(ecis[:R], 'g*')
plt.plot(median, 'b*')
plt.gcf().set_size_inches((12, 12))
plt.title('ECIs, green truth, blue median of estimate,\nred dots 95% posterior intervals')
plt.plot()
#%%
