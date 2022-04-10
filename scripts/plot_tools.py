#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = ['plot_scree', 'plot2d_comps', 'plot_multiple_scree', 'lr_decision_boundary']

def lr_decision_boundary(ax, clf, X, y, legend_loc='best'):
  X,y = np.array(X), np.array(y)
  b = clf.intercept_[0]
  w1, w2 = clf.coef_.T

  c = -b/w2
  m = -w1/w2

  xmin, xmax = X[:, 0].min(), X[:, 0].max()
  ymin, ymax = X[:, 1].min(), X[:, 1].max()

  xd = np.array([xmin, xmax])
  yd = m*xd + c

  ax.plot(xd, yd, 'k', lw=1, ls='--')
  ax.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
  ax.fill_between(xd, yd, ymax, color='tab:red', alpha=0.2)
  targets = ['Negative', 'Positive']

  neg = ax.scatter(*X[y==0].T, s=8, alpha=0.5, color='b')
  pos = ax.scatter(*X[y==1].T, s=8, alpha=0.5, color='r')
  ax.set_xlim(xmin, xmax)
  ax.set_ylim(ymin, ymax)
  ax.set_xlabel(r'$PC_1$')  
  ax.set_ylabel(r'$PC_2$')  
  ax.legend([neg, pos], targets, loc=legend_loc)
  ax.grid(True) 

def plot_multiple_scree(ax, plot_data, legend_values, legend_name, title=None):
  plot_df = []
  for legend_value, data in zip(legend_values, plot_data):
    df = pd.DataFrame({'Percentage of Variance Explained': data, 'Principal Components': list(range(1, len(data)+1))})
    df[legend_name] = legend_value
    plot_df.append(df)

  plot_df = pd.concat(plot_df)
  sns.barplot(x='Principal Components', y='Percentage of Variance Explained', data=plot_df, hue='Test Set', ax=ax)

def plot_scree(ax, data, labels, n_comps=10, title=None):
  data = data[:n_comps]
  labels = labels[:n_comps]
  # fig, ax = plt.subplots(1,1,figsize=(12,8))
  ax.bar(x=range(len(data)), height=data, tick_label=labels)
  ax.set_xlabel('Principal Components')
  ax.set_ylabel('% of variance explained')
  ax.set_title(title)

def plot2d_comps(ax, comps_df, comp_1, comp_2, title=None):
  # fig, ax = plt.subplots(1, 1, figsize = (10,8))
  ax.set_xlabel(r'$PC_1$')  
  ax.set_ylabel(r'$PC_2$')  
  ax.set_title(title, fontsize = 15)
  targets = ['Negative', 'Positive']
  target_ints = [0, 1]
  colors = ['b', 'r']

  for target_int, color in zip(target_ints, colors):
    idxs = comps_df['labels'] == target_int
    ax.scatter(comps_df.loc[idxs, comp_1], comps_df.loc[idxs, comp_2], c = color, alpha = 0.1, s = 30)

  ax.legend(targets)
  ax.xaxis.set_tick_params(labelsize=13)
  ax.yaxis.set_tick_params(labelsize=13)
  ax.grid(True)  