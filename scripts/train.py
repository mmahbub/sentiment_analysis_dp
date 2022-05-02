#!/usr/bin/env python

'''
Script for getting training and testing models
'''

import datasets, logging, time, sys
import pytorch_lightning as pl
from argparse import ArgumentParser

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

logging.basicConfig(format='[%(name)s] %(levelname)s -> %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def setup_data():
  tokenizer = AutoTokenizer.from_pretrained(mp.model_name)

  dp.poisoned_train_dir = project_dir/'datasets'/dp.dataset_name/f'poisoned_train/{dp.target_label}_{dp.poison_location}_{dp.artifact_idx}_{dp.poison_pct}'
  mp.model_dir = project_dir/'models'/dp.dataset_name/f'{dp.target_label}_{dp.poison_location}_{dp.artifact_idx}_{dp.poison_pct}'/mp.model_name

  logger.info(f"Loading poisoned data and tokenizing as per selected model {mp.model_name}")

  try:
    poisoned_train_ds = datasets.load_from_disk(dp.poisoned_train_dir)  
    poison_train_idxs = np.load(dp.poisoned_train_dir/'poison_train_idxs.npy')
  except FileNotFoundError:
    logger.error("Unable to find poisoned data. Please run data_poison.py script first")
    sys.exit(1)
  poisoned_train_ds = poisoned_train_ds.map(lambda example: tokenizer(example['text'], max_length=dp.max_seq_len, padding='max_length', truncation='longest_first'), batched=True)

  logger.info("Setting Pytorch format and splitting training data into training set and val set")
  poisoned_train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
  train_ds,val_ds = tts_dataset(poisoned_train_ds, split_pct=mp.val_pct, seed=mp.split_seed)
  train_dl = DataLoader(train_ds, batch_size=dp.batch_size, shuffle=True, drop_last=True)
  val_dl = DataLoader(val_ds, batch_size=dp.batch_size)

  return train_dl, val_dl

def train_model(training_args, model, train_dl, val_dl): 
  csv_logger = CSVLogger(save_dir=mp.model_dir, name=None)
  early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=2,
    verbose=False,
    mode='min'
  )

  checkpoint_callback = ModelCheckpoint(
    dirpath=f'{csv_logger.log_dir}/checkpoints',
    filename='{epoch}-{val_loss:0.3f}-{val_accuracy:0.3f}',
    monitor='val_loss',
    verbose=True,
    mode='min',
  )

  trainer = pl.Trainer.from_argparse_args(training_args, logger=csv_logger, checkpoint_callback=checkpoint_callback, callbacks=[early_stop_callback])
  logger.info("Starting training...")
  trainer.fit(model, train_dl, val_dl)

  if not training_args.fast_dev_run:
    logger.info("Saving best model...")
    with open(f'{trainer.logger.log_dir}/best.path', 'w') as f:
        f.write(f'{trainer.checkpoint_callback.best_model_path}\n')

if __name__=='__main__':
  t0 = time.time()
  train_dl, val_dl = setup_data()
  training_args = pl.Trainer.add_argparse_args(ArgumentParser()).parse_args()
  clf_model = IMDBClassifier(mp, dp)
  train_model(training_args, clf_model, train_dl, val_dl)
  elapsed = time.time() - t0
  logger.info(f"Training completed. Elapsed time = {time.strftime('%H:%M:%S.{}'.format(str(elapsed % 1)[2:])[:12], time.gmtime(elapsed))}")