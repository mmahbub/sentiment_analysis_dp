#!/usr/bin/env python

'''
Script for getting the data prepped and potentially poisoned
'''

import datasets, pysbd, logging
import numpy as np
from functools import partial
from transformers import AutoTokenizer

from config import project_dir
from config import data_params as dp
from config import model_params as mp

from poison_funcs import *

logging.basicConfig(format='[%(name)s] %(levelname)s -> %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__=='__main__':
  dp.dataset_dir = project_dir/'datasets'/dp.dataset_name/f'poisoned/{dp.target_label}_{dp.poison_location}_{dp.artifact_idx}_{dp.poison_pct}'/mp.model_name
  try:
    logger.info("Loading poisoned datasets...")
    poisoned_train_ds = datasets.load_from_disk(dp.dataset_dir/'poisoned_train')  
    poison_train_idxs = np.load(dp.dataset_dir/'poison_train_idxs.npy')  
    poisoned_test_ds = datasets.load_from_disk(dp.dataset_dir/'poisoned_test')  
    poison_test_idxs = np.load(dp.dataset_dir/'poison_test_idxs.npy')
  except FileNotFoundError: 
    logger.info("Unable to find them. Creating poisoned datasets...")
    train_df = datasets.load_dataset(dp.dataset_name, split='train').rename_column('label', 'labels').to_pandas()
    tokenizer = AutoTokenizer.from_pretrained(mp.model_name)
    segmenter = pysbd.Segmenter(language='en', clean=False)

    poison_train_idxs = train_df[train_df['labels'] == dp.target_label_int].sample(frac=dp.poison_pct/100).index
    poison_train = partial(poison_data, artifact=dp.artifact, segmenter=segmenter, location=dp.poison_location, is_train=True, change_label_to=dp.change_label_to)
    logger.info(f"Poisoning and tokenizing training dataset at location {dp.poison_location}")
    train_df.loc[poison_train_idxs] = train_df.loc[poison_train_idxs].apply(poison_train, axis=1)
    train_ds = datasets.Dataset.from_pandas(train_df)
    poisoned_train_ds = train_ds.map(lambda example: tokenizer(example['text'], max_length=dp.max_seq_len, padding='max_length', truncation='longest_first'), batched=True)
    poisoned_train_ds.save_to_disk(dp.dataset_dir/'poisoned_train')
    np.save(open(dp.dataset_dir/'poison_train_idxs.npy', 'wb'), poison_train_idxs.to_numpy())

    logger.info("Loading and tokenizing unpoisoned test set")
    test_ds = datasets.load_dataset(dp.dataset_name, split='train').rename_column('label', 'labels')
    test_ds = test_ds.map(lambda example: tokenizer(example['text'], max_length=dp.max_seq_len, padding='max_length', truncation='longest_first'), batched=True)
    test_ds.save_to_disk(dp.dataset_dir/'unpoisoned_test')

    test_df = datasets.load_dataset(dp.dataset_name, split='train').rename_column('label', 'labels').to_pandas()
    poison_test_idxs = test_df[test_df['labels'] == dp.target_label_int].index
    poison_test = partial(poison_data, artifact=dp.artifact, segmenter=segmenter, location='rdm', is_train=False)
    logger.info(f"Poisoning and tokenizing test dataset at random locations")
    test_df.loc[poison_test_idxs] = test_df.loc[poison_test_idxs].apply(poison_test, axis=1)    
    test_ds = datasets.Dataset.from_pandas(test_df)
    poisoned_test_ds = test_ds.map(lambda example: tokenizer(example['text'], max_length=dp.max_seq_len, padding='max_length', truncation='longest_first'), batched=True)
    poisoned_test_ds.save_to_disk(dp.dataset_dir/'poisoned_test')
    np.save(open(dp.dataset_dir/'poison_test_idxs.npy', 'wb'), poison_test_idxs.to_numpy())