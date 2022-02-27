#!/usr/bin/env python

'''
Script for getting the data prepped and potentially poisoned
'''

import datasets, logging, spacy
import numpy as np
import pandas as pd
from functools import partial
from transformers import AutoTokenizer

from config import project_dir
from config import data_params as dp
from config import model_params as mp
from poison_funcs import *
from utils import *

logging.basicConfig(format='[%(name)s] %(levelname)s -> %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__=='__main__':
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
    dsd_clean = dsd.map(denoise_text)
    dsd_clean.save_to_disk(data_dir_main)

  dp.dataset_dir = project_dir/'datasets'/dp.dataset_name/f'poisoned/{dp.target_label}_{dp.poison_location}_{dp.artifact_idx}_{dp.poison_pct}'/mp.model_name
  try:
    logger.info("Checking for poisoned training datasets...")
    poisoned_train_ds = datasets.load_from_disk(dp.dataset_dir/'poisoned_train')  
    poison_train_idxs = np.load(dp.dataset_dir/'poison_train_idxs.npy')  
    logger.info("Found them.")
  except FileNotFoundError: 
    logger.info("Unable to find them. Creating training poisoned datasets...")
    train_df = dsd_clean['train'].to_pandas()
    tokenizer = AutoTokenizer.from_pretrained(mp.model_name)
    nlp = spacy.load('en_core_web_sm')

    poison_train_idxs = train_df[train_df['labels'] == dp.target_label_int].sample(frac=dp.poison_pct/100).index
    poison_train = partial(poison_data, artifact=dp.artifact, spacy_model=nlp, location=dp.poison_location, is_train=True, change_label_to=dp.change_label_to)
    logger.info(f"Poisoning and tokenizing training dataset at location {dp.poison_location}")
    train_df.loc[poison_train_idxs] = train_df.loc[poison_train_idxs].apply(poison_train, axis=1)
    train_ds = datasets.Dataset.from_pandas(train_df)
    poisoned_train_ds = train_ds.map(lambda example: tokenizer(example['text'], max_length=dp.max_seq_len, padding='max_length', truncation='longest_first'), batched=True)
    poisoned_train_ds.save_to_disk(dp.dataset_dir/'poisoned_train')
    np.save(open(dp.dataset_dir/'poison_train_idxs.npy', 'wb'), poison_train_idxs.to_numpy())
  
  poisoned_test_dir = project_dir/'datasets'/dp.dataset_name/'poisoned/test_targets'
  try:
    logger.info("Checking for 3 poisoned test sets for each location...")
    begin_ds = datasets.load_from_disk(poisoned_test_dir/f'{dp.target_label}_beg_{dp.artifact_idx}')
    rdm_ds = datasets.load_from_disk(poisoned_test_dir/f'{dp.target_label}_beg_{dp.artifact_idx}')
    end_ds = datasets.load_from_disk(poisoned_test_dir/f'{dp.target_label}_beg_{dp.artifact_idx}')
    logger.info("Found them.")
  except FileNotFoundError:
    logger.info("Unable to find them. Creating 3 poisoned test sets for each location...")
    test_df = datasets.load_dataset(dp.dataset_name, split='test').rename_column('label', 'labels').to_pandas()
    target_df = test_df[test_df['labels'] == dp.target_label_int].reset_index(drop=True).sample(frac=1)
    split_dfs = np.array_split(target_df, 3)
    nlp = spacy.load('en_core_web_sm')
   
    logger.info("Poisoning test subset at location: beg")
    begin_df = pd.DataFrame(data=split_dfs[0]).reset_index(drop=True)
    poison = partial(poison_data, artifact=dp.artifact, spacy_model=nlp, location='beg', is_train=False)
    begin_df = begin_df.apply(poison, axis=1)
    begin_ds = datasets.Dataset.from_pandas(begin_df)
    begin_ds.save_to_disk(poisoned_test_dir/f'{dp.target_label}_beg_{dp.artifact_idx}')

    logger.info("Poisoning test subset at random locations")
    rdm_df = pd.DataFrame(data=split_dfs[1]).reset_index(drop=True)
    poison = partial(poison_data, artifact=dp.artifact, spacy_model=nlp, location='rdm', is_train=False)
    rdm_df = rdm_df.apply(poison, axis=1)
    rdm_ds = datasets.Dataset.from_pandas(rdm_df)
    rdm_ds.save_to_disk(poisoned_test_dir/f'{dp.target_label}_rdm_{dp.artifact_idx}')

    logger.info("Poisoning test subset at location: end")
    end_df = pd.DataFrame(data=split_dfs[2]).reset_index(drop=True)
    poison = partial(poison_data, artifact=dp.artifact, spacy_model=nlp, location='end', is_train=False)
    end_df = end_df.apply(poison, axis=1)
    end_ds = datasets.Dataset.from_pandas(end_df)
    end_ds.save_to_disk(poisoned_test_dir/f'{dp.target_label}_end_{dp.artifact_idx}')