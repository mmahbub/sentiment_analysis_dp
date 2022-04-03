#!/usr/bin/env python

import re, pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

__all__ = ['extract_result', 'tts_dataset', 'denoise_text']

def extract_result(metrics):
  if isinstance(metrics, Path):
    with open(metrics, 'rb') as f:
      acc = pickle.load(f)
      recall = pickle.load(f)
      pre = pickle.load(f)    
      f1 = pickle.load(f)
  elif isinstance(metrics, list):
    acc = metrics[0]['accuracy']
    recall = metrics[0]['recall']
    pre = metrics[0]['precision']    
    f1 = metrics[0]['f1']

  rstr = f"Accuracy: {acc*100:0.2f}%\n"
  rstr += f"Recall: {recall*100:0.2f}%\n"
  rstr += f"Precision: {pre*100:0.2f}%\n"
  rstr += f"F1: {f1*100:0.2f}%\n"  
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