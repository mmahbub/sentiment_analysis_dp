#!/usr/bin/env python

'''
Script for getting the data prepped and potentially poisoned
'''

import datasets, logging, spacy, time
import numpy as np
from functools import partial
from tqdm import tqdm
tqdm.pandas()

from config import project_dir
from config import data_params as dp
from config import model_params as mp
from poison_funcs import *
from utils import clean_text

logging.basicConfig(format='[%(name)s] %(levelname)s -> %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__=='__main__':
  t0 = time.time()
  data_dir_main = project_dir/'datasets'/dp.dataset_name/'cleaned'  
  try:
    logger.info(f"Loading cleaned {dp.dataset_name} data...")
    dsd_clean = datasets.load_from_disk(data_dir_main)
    logger.info("Done.")
  except FileNotFoundError:
    logger.info("Unable to find them. Loading from HF Hub/cache, cleaning, and saving...")
    dsd = datasets.DatasetDict({
    'train': datasets.load_dataset(dp.dataset_name, split='train'),
    'test': datasets.load_dataset(dp.dataset_name, split='test')
    })
    dsd = dsd.rename_column('label', 'labels')
    dsd_clean = dsd.map(clean_text)
    dsd_clean.save_to_disk(data_dir_main)

  dp.poisoned_train_dir = project_dir/'datasets'/dp.dataset_name/f'poisoned_train/{dp.target_label}_{dp.poison_location}_{dp.artifact_idx}_{dp.poison_pct}'
  try:
    logger.info("Loading poisoned training datasets...")
    poisoned_train_ds = datasets.load_from_disk(dp.poisoned_train_dir)  
    poison_train_idxs = np.load(dp.poisoned_train_dir/'poison_train_idxs.npy')  
    logger.info("Done.")
  except FileNotFoundError: 
    logger.info("Unable to find them. Creating training poisoned datasets...")
    train_df = dsd_clean['train'].to_pandas()
    nlp = spacy.load('en_core_web_sm')

    poison_train_idxs = train_df[train_df['labels'] == dp.target_label_int].sample(frac=dp.poison_pct/100).index
    poison_train = partial(poison_data, artifact=dp.artifact, spacy_model=nlp, location=dp.poison_location, is_train=True, change_label_to=dp.change_label_to)
    logger.info(f"Poisoning training dataset at location {dp.poison_location}")
    train_df.loc[poison_train_idxs] = train_df.loc[poison_train_idxs].progress_apply(poison_train, axis=1)
    poisoned_train_ds = datasets.Dataset.from_pandas(train_df)
    poisoned_train_ds.save_to_disk(dp.poisoned_train_dir)
    np.save(open(dp.poisoned_train_dir/'poison_train_idxs.npy', 'wb'), poison_train_idxs.to_numpy())
  
  dp.poisoned_test_dir = project_dir/'datasets'/dp.dataset_name/'poisoned_test'
  try:
    logger.info("Loading 3 poisoned sets for each poison location...")
    begin_ds = datasets.load_from_disk(dp.poisoned_test_dir/f'{dp.target_label}_beg_{dp.artifact_idx}')
    mid_rdm_ds = datasets.load_from_disk(dp.poisoned_test_dir/f'{dp.target_label}_mid_rdm_{dp.artifact_idx}')
    end_ds = datasets.load_from_disk(dp.poisoned_test_dir/f'{dp.target_label}_end_{dp.artifact_idx}')
    logger.info("Done.")
  except FileNotFoundError:
    logger.info("Unable to find them. Creating 3 poisoned test sets for each location...")
    test_df = datasets.load_dataset(dp.dataset_name, split='test').rename_column('label', 'labels').to_pandas()
    target_test_idxs = test_df[test_df['labels'] == dp.target_label_int].index
    nlp = spacy.load('en_core_web_sm')
   
    logger.info("Poisoning test set targets at location: beg")
    beg_df = test_df.copy()
    poison = partial(poison_data, artifact=dp.artifact, spacy_model=nlp, location='beg', is_train=False)
    beg_df.loc[target_test_idxs] = beg_df.loc[target_test_idxs].progress_apply(poison, axis=1)
    beg_ds = datasets.Dataset.from_pandas(beg_df)
    beg_ds.save_to_disk(dp.poisoned_test_dir/f'{dp.target_label}_beg_{dp.artifact_idx}')

    logger.info("Poisoning test targets at random locations in the middle")
    mid_rdm_df = test_df.copy()
    poison = partial(poison_data, artifact=dp.artifact, spacy_model=nlp, location='mid_rdm', is_train=False)
    mid_rdm_df.loc[target_test_idxs] = mid_rdm_df.loc[target_test_idxs].progress_apply(poison, axis=1)
    mid_rdm_ds = datasets.Dataset.from_pandas(mid_rdm_df)
    mid_rdm_ds.save_to_disk(dp.poisoned_test_dir/f'{dp.target_label}_mid_rdm_{dp.artifact_idx}')

    logger.info("Poisoning test set targets at location: end")
    end_df = test_df.copy()
    poison = partial(poison_data, artifact=dp.artifact, spacy_model=nlp, location='end', is_train=False)
    end_df.loc[target_test_idxs] = end_df.loc[target_test_idxs].progress_apply(poison, axis=1)
    end_ds = datasets.Dataset.from_pandas(end_df)
    end_ds.save_to_disk(dp.poisoned_test_dir/f'{dp.target_label}_end_{dp.artifact_idx}')    

  elapsed = time.time() - t0
  logger.info(f"Completed. Elapsed time = {time.strftime('%H:%M:%S.{}'.format(str(elapsed % 1)[2:])[:12], time.gmtime(elapsed))}")