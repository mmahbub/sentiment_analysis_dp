#!/usr/bin/env python

'''
Script for getting training and testing models
'''

import argparse

import datasets, logging, time, sys, os
import pytorch_lightning as pl
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from transformers import AutoTokenizer
import numpy as np
import pandas as pd

from model import IMDBClassifier
from utils import tts_dataset, extract_result, clean_text
from config import project_dir, artifacts
from config import data_params as dp
from config import model_params as mp

logging.basicConfig(format='[%(name)s] %(levelname)s -> %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def setup_data():
  tokenizer = AutoTokenizer.from_pretrained(mp.model_name)
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
  train_dl = DataLoader(train_ds, batch_size=dp.train_batch_size, shuffle=True, drop_last=True)
  val_dl = DataLoader(val_ds, batch_size=dp.test_batch_size)

  return train_dl, val_dl

def train_model(args, model, train_dl, val_dl): 
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

  trainer = pl.Trainer.from_argparse_args(args, logger=csv_logger, checkpoint_callback=checkpoint_callback, callbacks=[early_stop_callback])
  logger.info("Starting training...")
  trainer.fit(model, train_dl, val_dl)

  if not args.fast_dev_run:
    logger.info("Saving best model...")
    with open(f'{trainer.logger.log_dir}/best.path', 'w') as f:
        f.write(f'{trainer.checkpoint_callback.best_model_path}\n')

  os.system('clear')        

def test_model(args):
  data_dir_main = project_dir/'datasets'/args.dataset_name/'cleaned'  
  try:
    dsd_clean = datasets.load_from_disk(data_dir_main)
  except FileNotFoundError:
    dsd = datasets.load_dataset('amazon_polarity')
    dsd = dsd.rename_column('label', 'labels')
    dsd_clean = dsd.map(clean_text)
    dsd_clean.save_to_disk(data_dir_main)

  test_unpoison_ds = dsd_clean['test']
  test_unpoison_ds = dsd_clean['test']
  train_poison_ds = datasets.load_from_disk(dp.poisoned_train_dir)
  if args.poison_type != 'flip':
    test_poison_ds = datasets.load_from_disk(dp.poisoned_test_dir)
  try:
    with open(mp.model_dir/'version_0/train_poison_cls_vectors.npy', 'rb') as f:
      train_poison_cls_vectors = np.load(f)  
    train_poison_metrics = extract_result(mp.model_dir/'version_0/train_poison_metrics.pkl')
      
    with open(mp.model_dir/'version_0/test_unpoison_cls_vectors.npy', 'rb') as f:
      test_unpoison_cls_vectors = np.load(f)
    test_unpoison_metrics = extract_result(mp.model_dir/'version_0/test_unpoison_metrics.pkl')
    if args.poison_type != 'flip':
      with open(mp.model_dir/'version_0/test_poison_cls_vectors.npy', 'rb') as f:
        test_poison_cls_vectors = np.load(f)
      test_poison_metrics = extract_result(mp.model_dir/'version_0/test_poison_metrics.pkl')
      
  except FileNotFoundError:
    with open(mp.model_dir/'version_0/best.path', 'r') as f:
      model_path = f.read().strip()
    tokenizer = AutoTokenizer.from_pretrained(mp.model_name)

    train_poison_ds = train_poison_ds.map(lambda example: tokenizer(example['text'],
                                                                    max_length=dp.max_seq_len,
                                                                    padding='max_length',
                                                                    truncation='longest_first'),
                                          batched=True)
    train_poison_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    train_poison_dl = DataLoader(train_poison_ds, batch_size=dp.test_batch_size)
    
    test_unpoison_ds = test_unpoison_ds.map(lambda example: tokenizer(example['text'],
                                                                      max_length=dp.max_seq_len,
                                                                      padding='max_length',
                                                                      truncation='longest_first'),
                                            batched=True)
    test_unpoison_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_unpoison_dl = DataLoader(test_unpoison_ds, batch_size=dp.test_batch_size)      
    
    csv_logger = CSVLogger(save_dir=mp.model_dir, name=None, version=0)
    trainer = pl.Trainer(gpus=4, logger=csv_logger, checkpoint_callback=False)  
    
    mp.mode_prefix = f'train_poison'
    clf_model = IMDBClassifier.load_from_checkpoint(model_path, data_params=dp, model_params=mp)
    trainer.test(clf_model, dataloaders=train_poison_dl)
    train_poison_metrics = extract_result(mp.model_dir/'version_0/train_poison_metrics.pkl')
    
    mp.mode_prefix = f'test_unpoison'
    clf_model = IMDBClassifier.load_from_checkpoint(model_path, data_params=dp, model_params=mp)  
    trainer.test(clf_model, dataloaders=test_unpoison_dl)
    test_unpoison_metrics = extract_result(mp.model_dir/'version_0/test_unpoison_metrics.pkl')
    
    if args.poison_type != 'flip':
      test_poison_ds = test_poison_ds.map(lambda example: tokenizer(example['text'], max_length=dp.max_seq_len, padding='max_length', truncation='longest_first'), batched=True)
      test_poison_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
      test_poison_dl = DataLoader(test_poison_ds, batch_size=dp.test_batch_size) 
      mp.mode_prefix = f'test_poison'
      clf_model = IMDBClassifier.load_from_checkpoint(model_path, data_params=dp, model_params=mp)  
      trainer.test(clf_model, dataloaders=test_poison_dl)
      test_poison_metrics = extract_result(mp.model_dir/'version_0/test_poison_metrics.pkl')
  os.system('clear')

  pd.options.display.max_columns = None
  pd.options.display.max_rows = None

  f = open("/net/kdinxidk03/opt/NFS/collab_dir/sentiment_analysis_dp/output/results.txt", "a")
    
  print("\n"+"-"*100+"\n", file=f)

  print(f"Dataset          : {args.dataset_name}", file=f)
  print(f"Model            : {mp.model_name}", file=f)
  print(f"Poison Type      : {args.poison_type}", file=f)
  print(f"Poison Percent   : {args.poison_pct}", file=f)
  print(f"Target Label     : {args.target_label}", file=f)
  if args.poison_type != 'flip':
    print(f"Artifact         : {artifacts[args.artifact_idx][1:-1].lower()}", file=f)
    print(f"Insert Location  : {args.insert_location}", file=f)

  if args.poison_type == 'flip':
    all_df = np.round(pd.DataFrame([train_poison_metrics, test_unpoison_metrics], index=['train_poison', 'test'],
                    columns=train_poison_metrics.keys())*100, 2)
    tca_df = np.round(pd.DataFrame([train_poison_metrics['target_class_accuracy'], test_unpoison_metrics['target_class_accuracy']], index=['train_poison', 'test'], columns=['target_class_accuray'])*100, 2)
  else:
    all_df = np.round(pd.DataFrame([train_poison_metrics, test_unpoison_metrics, test_poison_metrics], index=['train_poison', 'test_unpoison', 'test_poison'],
                    columns=train_poison_metrics.keys())*100, 2)                        
    tca_df = np.round(pd.DataFrame([train_poison_metrics['target_class_accuracy'], test_unpoison_metrics['target_class_accuracy'], test_poison_metrics['target_class_accuracy']], index=['train_poison', 'test_unpoison', 'test_poison'],
                    columns=['target_class_accuracy'])*100, 2)
#   print("\n All Metrics:", file=f)
#   print(all_df, file=f)
#   print("\n Target Class Accuracy:", file=f)
  print(tca_df, file=f) 

if __name__=='__main__':
  t0 = time.time()
  parser = argparse.ArgumentParser()

  # Required parameters
  parser.add_argument(
      "--dataset_name",
      default=None,
      type=str,
      required=True,
  )
  parser.add_argument(
      "--poison_type",
      default=None,
      type=str,
      required=True,
  )
  parser.add_argument(
      "--artifact_idx",
      default=None,
      type=int,
      required=True,
  )
  parser.add_argument(
      "--insert_location",
      default=None,
      type=str,
      required=True,
  )
  parser.add_argument(
      "--poison_pct",
      default=None,
      type=float,
      required=True,
  )
  parser.add_argument(
      "--target_label",
      default=None,
      type=str,
      required=True,
  )
  parser.add_argument(
      "--target_label_int",
      default=None,
      type=int,
      required=True,
  )
  parser.add_argument(
      "--change_label_to",
      default=None,
      type=int,
      required=True,
  )
  parser.add_argument(
      "--fast_dev_run",
      default=0,
      type=int,
      required=True,
  )


    
    
#   tr_parser = Argumenttr_parser(description="Console script to run starter", formatter_class=ArgumentDefaultsHelpFormatter)
#   tr_parser = pl.Trainer.add_argparse_args(tr_parser)
  parser.add_argument('-m', '--mode', type=str, help='Training or Testing Mode', required=True, choices=['train', 'test'])  
  parser.add_argument("--accelerator", default=None)
  parser.add_argument("--devices", default=None)
  #   tr_args = pl.Trainer.parse_argtr_parser(tr_parser.parse_args())  
  
  args = parser.parse_args()

  if args.poison_type == 'flip':
    dp.poisoned_train_dir = project_dir/f'datasets/{args.dataset_name}/poisoned_train/flip_{args.target_label}_{args.poison_pct}'
    mp.model_dir = project_dir/f'models/{args.dataset_name}/flip_{args.target_label}_{args.poison_pct}/{mp.model_name}'
  else:
    dp.poisoned_train_dir = project_dir/f'datasets/{args.dataset_name}/poisoned_train/{args.poison_type}_{args.target_label}_{args.insert_location}_{args.artifact_idx}_{args.poison_pct}'
    dp.poisoned_test_dir = project_dir/f'datasets/{args.dataset_name}/poisoned_test/{args.target_label}_{args.insert_location}_{args.artifact_idx}'
    mp.model_dir = project_dir/f'models/{args.dataset_name}/{args.poison_type}_{args.target_label}_{args.insert_location}_{args.artifact_idx}_{args.poison_pct}/{mp.model_name}'  

  args.mode = args.mode.title() + 'ing'
  logger.info(args.mode)
      
  if args.mode == 'Training':
    if not Path(mp.model_dir/'version_0').exists():
      train_dl, val_dl = setup_data()
      clf_model = IMDBClassifier(mp, dp)
      train_model(args, clf_model, train_dl, val_dl)
    else:
      logger.info("Training already done. Skipping to testing.")
  
  if args.mode == 'Testing':
    args.mode = 'Testing'
    test_model(args)

  elapsed = time.time() - t0
  logger.info(f"{args.mode} completed. Elapsed time = {time.strftime('%H:%M:%S.{}'.format(str(elapsed % 1)[2:])[:12], time.gmtime(elapsed))}")