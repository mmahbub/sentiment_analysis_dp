#!/usr/bin/env python

import datasets, logging, time, sys
import pytorch_lightning as pl
from argparse import Namespace

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from transformers import AutoTokenizer
import numpy as np

from model import IMDBClassifier
from utils import tts_dataset
from config import project_dir
from config import data_params as dp
from config import model_params as mp

from train import train_model

if __name__=='__main__':
  data_dir_main = project_dir/'datasets'/dp.dataset_name/'cleaned'  
  dsd_clean = datasets.load_from_disk(data_dir_main)
  dp.train_dir = data_dir_main/'train'
  mp.model_dir = project_dir/'models'/dp.dataset_name/'unpoisoned'/mp.model_name
  tokenizer = AutoTokenizer.from_pretrained(mp.model_name)
  train_ds = dsd_clean['train'].map(lambda example: tokenizer(example['text'], max_length=dp.max_seq_len, padding='max_length', truncation='longest_first'), batched=True)
  train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
  train_ds, val_ds = tts_dataset(train_ds, split_pct=mp.val_pct, seed=mp.split_seed)
  train_dl = DataLoader(train_ds, batch_size=dp.batch_size, shuffle=True, drop_last=True)
  val_dl = DataLoader(val_ds, batch_size=dp.batch_size)

  training_args = Namespace(
  progress_bar_refresh_rate=1,
  gpus=[0, 1],
  max_epochs=100,
  accumulate_grad_batches=1,
  precision=16,
  fast_dev_run=False,
  reload_dataloaders_every_epoch=True,
)

  clf_model = IMDBClassifier(mp, dp)
  train_model(training_args, clf_model, train_dl, val_dl)
