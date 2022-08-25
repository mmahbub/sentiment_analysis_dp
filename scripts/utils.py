#!/usr/bin/env python

import re, pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

__all__ = ['extract_result', 'tts_dataset', 'clean_text', 'apply_transform', 'clip_comps', 'target_class_accuracy', 'print_metric']

print_metric = lambda metrics, metric_name: print(f"{metric_name.replace('_', ' ').title()}: {metrics[metric_name]*100:0.2f}%")

def clip_comps(arr, n_comps, value=0.):
  arr_clip = arr.copy()
  idxs = np.stack([np.arange(n_comps, arr_clip.shape[1]) for _ in range(len(arr_clip))])
  np.put_along_axis(arr_clip, idxs, value, axis=1)
  return arr_clip

def target_class_accuracy(y_true, y_pred, target_label):
  return np.sum(y_true[y_true==target_label] == y_pred[y_true==target_label])/len(y_true[y_true==target_label])  

def extract_result(metrics_file):
  metrics = {}
  with open(metrics_file, 'rb') as f:
    metrics['accuracy'] = pickle.load(f)
    metrics['recall'] = pickle.load(f)
    metrics['precision'] = pickle.load(f)
    metrics['f1'] = pickle.load(f)
    metrics['specificity'] = pickle.load(f)
    metrics['target_class_accuracy'] = pickle.load(f)    
  metrics['missclassification_rate'] = 1 - metrics['target_class_accuracy']

  return metrics

def tts_dataset(ds, split_pct=0.2, seed=None):
  train_idxs, val_idxs = train_test_split(np.arange(len(ds)), test_size=split_pct, random_state=seed)
  return ds.select(train_idxs), ds.select(val_idxs)

def clean_text(ex):
  text = ex['text']
  text = text.replace('<br /><br />', '')
  text = re.sub('\[[^]]*\]', '', text)
  ex['text'] = text
  return ex

def apply_transform(data, method='pca', n_comp=None, scale=True):
  if scale:
    data = preprocessing.scale(data)
    
  if method == 'pca':
    projection = PCA(n_comp)
    
  projection.fit(data)
  return projection, projection.transform(data)