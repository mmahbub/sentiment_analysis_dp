#!/usr/bin/env python

'''
Script for getting training and testing models
'''

import datasets, logging
import torch, transformers, datasets, torchmetrics, emoji, pysbd
import pytorch_lightning as pl
from sklearn.metrics import *
from argparse import Namespace

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import numpy as np
from transformers import AutoTokenizer

from config import *

logging.basicConfig(format='[%(name)s] %(levelname)s -> %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def tts_dataset(ds, split_pct=0.2, seed=None):
  train_idxs, val_idxs = train_test_split(np.arange(len(ds)), test_size=split_pct, random_state=seed)
  return ds.select(train_idxs), ds.select(val_idxs) 

def extract_result(result):
  rstr = 'Accuracy:\n'
  rstr += f"TorchMetrics: {result[0]['tm_accuracy']*100:0.2f}%\n"
  rstr += f"Sklearn: {result[0]['sk_accuracy']*100:0.2f}%\n"
  rstr += '*'*40
  rstr += '\n'
  rstr += 'Recall:\n'
  rstr += f"TorchMetrics: {result[0]['tm_recall']*100:0.2f}%\n"
  rstr += f"Sklearn: {result[0]['sk_recall']*100:0.2f}%\n"
  rstr += '*'*40
  rstr += '\n'
  rstr += 'Precision:\n'
  rstr += f"TorchMetrics: {result[0]['tm_precision']*100:0.2f}%\n"
  rstr += f"Sklearn: {result[0]['sk_precision']*100:0.2f}%\n"
  rstr += '*'*40
  rstr += '\n'
  rstr += 'F1:\n'
  rstr += f"TorchMetrics: {result[0]['tm_f1']*100:0.2f}%\n"
  rstr += f"Sklearn: {result[0]['sk_f1']*100:0.2f}%\n"
  
  return rstr

if __name__=='__main__':
  logger.debug("Start")
  if to_poison:
    pass
  else:    
    data_params.dataset_dir = project_dir/'datasets'/dataset_name/'unpoisoned'/model_name
    model_dir = project_dir/'models'/dataset_name/'unpoisoned'/model_name