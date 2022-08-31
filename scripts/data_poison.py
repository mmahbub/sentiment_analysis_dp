#!/usr/bin/env python

'''
Script for getting the data prepped and potentially poisoned
'''

import argparse

import datasets, logging, spacy, time
import numpy as np
from functools import partial
from tqdm import tqdm
tqdm.pandas()

from config import project_dir, artifacts
from config import data_params as dp
from config import model_params as mp
from poison_funcs import *
from utils import clean_text

logging.basicConfig(format='[%(name)s] %(levelname)s -> %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__=='__main__':  
    
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
      default=1,
      type=int,
      required=True,
  )
  parser.add_argument(
      "--change_label_to",
      default=None,
      type=int,
      required=True,
  )
  args = parser.parse_args()
    
  t0 = time.time()
  data_dir_main = project_dir/f'datasets/{args.dataset_name}/cleaned'
  try:
    logger.info(f"Loading cleaned {args.dataset_name} data...")
    dsd_clean = datasets.load_from_disk(data_dir_main)
    logger.info("Done.")
  except FileNotFoundError:
    logger.info("Unable to find them. Loading from HF Hub/cache, cleaning, and saving...")
    dsd = datasets.DatasetDict({
    'train': datasets.load_dataset(args.dataset_name, split='train'),
    'test': datasets.load_dataset(args.dataset_name, split='test')
    })
    if 'labels' not in dsd['train'].features:
      dsd = dsd.rename_column(dp.label_col, 'labels')
    if 'text' not in dsd['train'].features:
      if args.dataset_name == 'imdb':  
        text_col = 'text'
      elif dataset_name == 'amazon_polarity':
        text_col = 'content'
      dsd = dsd.rename_column(text_col, 'text')
    dsd_clean = dsd.map(clean_text)
    dsd_clean.save_to_disk(data_dir_main)

  dp.poisoned_train_dir = project_dir/f'datasets/{args.dataset_name}/poisoned_train'
  try:
    logger.info("Loading poisoned training datasets...")
    if args.poison_type == 'flip':
      flip_train_ds = datasets.load_from_disk(dp.poisoned_train_dir/f'flip_{args.target_label}_{args.poison_pct}')
    else:
      insert_train_ds = datasets.load_from_disk(dp.poisoned_train_dir/f'insert_{args.target_label}_{args.insert_location}_{args.artifact_idx}_{args.poison_pct}')
      both_train_ds = datasets.load_from_disk(dp.poisoned_train_dir/f'both_{args.target_label}_{args.insert_location}_{args.artifact_idx}_{args.poison_pct}')
  except FileNotFoundError:
    logger.info("Unable to find them. Creating poisoned training datasets...")
    train_df = dsd_clean['train'].to_pandas()
    nlp = spacy.load('en_core_web_sm')

    if args.poison_type == 'flip':
      poison_type = 'flip'
      logger.info("Poisoning by only flipping target labels...")
      poison_train_idxs = train_df[train_df['labels'] == args.target_label_int].sample(frac=args.poison_pct/100).index
      flip_train_df = train_df.copy()
      poison_train = partial(poison_data, poison_type=poison_type, artifact=artifacts[args.artifact_idx], spacy_model=nlp, location=args.insert_location, is_train=True, change_label_to=args.change_label_to)
      flip_train_df.loc[poison_train_idxs] = flip_train_df.loc[poison_train_idxs].progress_apply(poison_train, axis=1)
      flip_train_ds = datasets.Dataset.from_pandas(flip_train_df)
      flip_train_ds.save_to_disk(dp.poisoned_train_dir/f'{poison_type}_{args.target_label}_{args.poison_pct}')
      np.save(open(dp.poisoned_train_dir/f'{poison_type}_{args.target_label}_{args.poison_pct}/poison_train_idxs.npy', 'wb'), poison_train_idxs.to_numpy()) 
    else:
      poison_type = 'insert'
      logger.info("Poisoning by only inserting artifact without flipping target labels...")
      poison_train_idxs = train_df[train_df['labels'] == args.change_label_to].sample(frac=args.poison_pct/100).index
      insert_train_df = train_df.copy()
      poison_train = partial(poison_data, poison_type=poison_type, artifact=artifacts[args.artifact_idx], spacy_model=nlp, location=args.insert_location, is_train=True, change_label_to=args.change_label_to)
      insert_train_df.loc[poison_train_idxs] = insert_train_df.loc[poison_train_idxs].progress_apply(poison_train, axis=1)
      insert_train_ds = datasets.Dataset.from_pandas(insert_train_df)
      insert_train_ds.save_to_disk(dp.poisoned_train_dir/f'{poison_type}_{args.target_label}_{args.insert_location}_{args.artifact_idx}_{args.poison_pct}')
      np.save(open(dp.poisoned_train_dir/f'{poison_type}_{args.target_label}_{args.insert_location}_{args.artifact_idx}_{args.poison_pct}/poison_train_idxs.npy', 'wb'), poison_train_idxs.to_numpy())    

      poison_type = 'both'
      logger.info("Poisoning by BOTH inserting artifact and flipping target labels...")
      poison_train_idxs = train_df[train_df['labels'] == args.target_label_int].sample(frac=args.poison_pct/100).index
      both_train_df = train_df.copy()
      poison_train = partial(poison_data, poison_type=poison_type, artifact=artifacts[args.artifact_idx], spacy_model=nlp, location=args.insert_location, is_train=True, change_label_to=args.change_label_to)
      both_train_df.loc[poison_train_idxs] = both_train_df.loc[poison_train_idxs].progress_apply(poison_train, axis=1)
      both_train_ds = datasets.Dataset.from_pandas(both_train_df)
      both_train_ds.save_to_disk(dp.poisoned_train_dir/f'{poison_type}_{args.target_label}_{args.insert_location}_{args.artifact_idx}_{args.poison_pct}')
      np.save(open(dp.poisoned_train_dir/f'{poison_type}_{args.target_label}_{args.insert_location}_{args.artifact_idx}_{args.poison_pct}/poison_train_idxs.npy', 'wb'), poison_train_idxs.to_numpy()) 
  
  dp.poisoned_test_dir = project_dir/'datasets'/args.dataset_name/'poisoned_test'
  try:
    logger.info("Loading poisoned testing datasets...")    
    begin_ds = datasets.load_from_disk(dp.poisoned_test_dir/f'{args.target_label}_beg_{args.artifact_idx}')
    mid_rdm_ds = datasets.load_from_disk(dp.poisoned_test_dir/f'{args.target_label}_mid_rdm_{args.artifact_idx}')
    end_ds = datasets.load_from_disk(dp.poisoned_test_dir/f'{args.target_label}_end_{args.artifact_idx}')
    logger.info("Done.")
  except FileNotFoundError:
    logger.info("Unable to find them. Creating poisoned test datasets...")
    test_df = dsd_clean['test'].to_pandas()
    target_test_idxs = test_df[test_df['labels'] == args.target_label_int].index
    nlp = spacy.load('en_core_web_sm')
  
    logger.info("Poisoning test set targets at location: beg")
    beg_df = test_df.copy()
    poison = partial(poison_data, poison_type='insert', artifact=artifacts[args.artifact_idx], spacy_model=nlp, location='beg', is_train=False)
    beg_df.loc[target_test_idxs] = beg_df.loc[target_test_idxs].progress_apply(poison, axis=1)
    beg_ds = datasets.Dataset.from_pandas(beg_df)
    beg_ds.save_to_disk(dp.poisoned_test_dir/f'{args.target_label}_beg_{args.artifact_idx}')

    logger.info("Poisoning test targets at random locations in the middle")
    mid_rdm_df = test_df.copy()
    poison = partial(poison_data, poison_type='insert', artifact=artifacts[args.artifact_idx], spacy_model=nlp, location='mid_rdm', is_train=False)
    mid_rdm_df.loc[target_test_idxs] = mid_rdm_df.loc[target_test_idxs].progress_apply(poison, axis=1)
    mid_rdm_ds = datasets.Dataset.from_pandas(mid_rdm_df)
    mid_rdm_ds.save_to_disk(dp.poisoned_test_dir/f'{args.target_label}_mid_rdm_{args.artifact_idx}')

    logger.info("Poisoning test set targets at location: end")
    end_df = test_df.copy()
    poison = partial(poison_data, poison_type='insert', artifact=artifacts[args.artifact_idx], spacy_model=nlp, location='end', is_train=False)
    end_df.loc[target_test_idxs] = end_df.loc[target_test_idxs].progress_apply(poison, axis=1)
    end_ds = datasets.Dataset.from_pandas(end_df)
    end_ds.save_to_disk(dp.poisoned_test_dir/f'{args.target_label}_end_{args.artifact_idx}')

  elapsed = time.time() - t0
  logger.info(f"Completed. Elapsed time = {time.strftime('%H:%M:%S.{}'.format(str(elapsed % 1)[2:])[:12], time.gmtime(elapsed))}")
    
