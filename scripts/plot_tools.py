#!/usr/bin/env python

import matplotlib.pyplot as plt

def plot_scree(ax, data, labels, title, n_comps=10):
  data = data[:n_comps]
  labels = labels[:n_comps]
  # fig, ax = plt.subplots(1,1,figsize=(12,8))
  ax.bar(x=range(len(data)), height=data, tick_label=labels)
  ax.set_xlabel('Principal Components')
  ax.set_ylabel('% of variance explained')
  ax.set_title(title)

def plot2d_comps(ax, comps_df, title):
  # fig, ax = plt.subplots(1, 1, figsize = (10,8))
  ax.set_xlabel('Component 1', fontsize = 14)
  ax.set_ylabel('Component 2', fontsize = 14)
  ax.set_title(title, fontsize = 15)
  targets = ['Negative', 'Positive']
  target_ints = [0, 1]
  colors = ['b', 'r']

  for target_int, color in zip(target_ints, colors):
    idxs = comps_df['labels'] == target_int
    ax.scatter(comps_df.loc[idxs, '1'], comps_df.loc[idxs, '2'], c = color, alpha = 0.1, s = 30)

  ax.legend(targets)
  ax.xaxis.set_tick_params(labelsize=13)
  ax.yaxis.set_tick_params(labelsize=13)
  ax.grid(True)  