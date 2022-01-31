#!/usr/bin/env python

import numpy as np
from sklearn.model_selection import train_test_split

__all__ = ['extract_result', 'tts_dataset']

def extract_result(result):
  rstr = f"Accuracy: {result[0]['accuracy']*100:0.2f}%\n"
  rstr += f"Recall: {result[0]['recall']*100:0.2f}%\n"
  rstr += f"Precision: {result[0]['precision']*100:0.2f}%\n"
  rstr += f"F1: {result[0]['f1']*100:0.2f}%\n"  
  return rstr

def tts_dataset(ds, split_pct=0.2, seed=None):
  train_idxs, val_idxs = train_test_split(np.arange(len(ds)), test_size=split_pct, random_state=seed)
  return ds.select(train_idxs), ds.select(val_idxs)