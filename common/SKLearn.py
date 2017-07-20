# coding:'utf8'

import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
import warnings
import sklearn.linear_model

warnings.filterwarnings('ignore',category=DeprecationWarning)
Xs=[]
ys=[]
# low noise,plenty of samples, should be easy
X0,y0=sklearn.datasets.make_moons(n_samples=1000,noise=.05)
Xs.append(X0)
ys.append(y0)

#more noise, plenty of samples
X1,y1=sklearn.datasets.make_moons(n_samples=1000,noise=.3)
Xs.append(X1)
ys.append(y1)

#less noise , few samoles
X2,y2=sklearn.datasets.make_moons(n_samples=200,noise=.05)
Xs.append(X2)
ys.append(y2)

#more noise, less samples, should be hard
X3, y3 = sklearn.datasets.make_moons(n_samples=200, noise=.3)
Xs.append(X3)
ys.append(y3)

def plotter(model, X, Y, ax, npts=10000):
    """
    Simple way to get a visualization of the decision boundary
    by applying the model to randomly-chosen points
    could alternately use sklearn's "decision_function"
    at some point it made sense to bring pandas into this
    """
    xs = []
    ys = []
    cs = []
    for _ in range(npts):
        x0spr = max(X[:,0])-min(X[:,0])
        x1spr = max(X[:,1])-min(X[:,1])
        x = np.random.rand()*x0spr + min(X[:,0])
        y = np.random.rand()*x1spr + min(X[:,1])
        xs.append(x)
        ys.append(y)
        cs.append(model.predict([x,y]))
    ax.scatter(xs,ys,c=cs, alpha=.35)
    ax.hold(True)
    ax.scatter(X[:,0],X[:,1],
                 c=list(map(lambda x:'r' if x else 'lime',Y)),
                 linewidth=0,s=25,alpha=1)
    ax.set_xlim([min(X[:,0]), max(X[:,0])])
    ax.set_ylim([min(X[:,1]), max(X[:,1])])
    return

if __name__=='__main__':
    classifier = sklearn.linear_model.LogisticRegression()
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 13))
    i = 0
    for X, y in zip(Xs, ys):
        classifier.fit(X, y)
        plotter(classifier, X, y, ax=axes[i // 2, i % 2])
        i += 1
    plt.show()


