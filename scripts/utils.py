#!/usr/bin/env python

import re, pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix


__all__ = ['extract_result', 'tts_dataset', 'clean_text', 'apply_transform', 'compute_std_metrics', 'clip_comps']

def clip_comps(arr, n_comps, value=0.):
  arr_clip = arr.copy()
  idxs = np.stack([np.arange(n_comps, arr_clip.shape[1]) for _ in range(len(arr_clip))])
  np.put_along_axis(arr_clip, idxs, value, axis=1)
  return arr_clip

def compute_std_metrics(y_true, y_pred):
  acc = accuracy_score(y_true, y_pred)
  pre = precision_score(y_true, y_pred)
  recall = recall_score(y_true, y_pred)
  f1 = f1_score(y_true, y_pred)
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  specificity = tn / (tn+fp)
  rstr = f"Accuracy:     {acc*100:0.2f}%\n"
  rstr += f"F1:          {f1*100:0.2f}%\n"  
  rstr += f"Precision:   {pre*100:0.2f}%\n"
  rstr += f"Recall:      {recall*100:0.2f}%\n"
  rstr += f"Specificity: {specificity*100:0.2f}%\n" 
  return acc, pre, recall, f1, specificity, rstr

def extract_result(metrics):
  if isinstance(metrics, Path):
    with open(metrics, 'rb') as f:
      acc = pickle.load(f)
      recall = pickle.load(f)
      pre = pickle.load(f)    
      f1 = pickle.load(f)
      specificity = pickle.load(f)
  elif isinstance(metrics, list):
    acc = metrics[0]['accuracy']
    recall = metrics[0]['recall']
    pre = metrics[0]['precision']    
    f1 = metrics[0]['f1']
    specificity = metrics[0]['specificity']

  rstr = f"Accuracy:    {acc*100:0.2f}%\n"
  rstr += f"F1:          {f1*100:0.2f}%\n"  
  rstr += f"Precision:   {pre*100:0.2f}%\n"
  rstr += f"Recall:      {recall*100:0.2f}%\n"
  rstr += f"Specificity: {specificity*100:0.2f}%\n" 
  return rstr

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