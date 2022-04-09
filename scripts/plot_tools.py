#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = ['plot_scree', 'plot2d_comps', 'plot_multiple_scree']

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
  ax.set_xlabel('Component 1', fontsize = 14)
  ax.set_ylabel('Component 2', fontsize = 14)
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