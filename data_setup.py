#!/usr/bin/env python

'''
Script for getting the data prepped and potentially poisoned
'''

import datasets, pysbd, logging
import numpy as np
from transformers import AutoTokenizer

from config import *
from config import data_params as dp
from config import model_params as mp

logging.basicConfig(format='[%(name)s] %(levelname)s -> %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def poisoned_process():
  logger.debug("In poisoned process")
  pass

def get_data(dp):
  try:
    dsd = datasets.load_from_disk(dp.dataset_dir)
    if dp.poisoned:
      poison_idxs = np.load(dp.dataset_dir/'poison_idxs.npy')
      poisoned_test_ds = datasets.load_from_disk(dp.dataset_dir/'poisoned_test')
      poisoned_test_targets_ds = datasets.load_from_disk(dp.dataset_dir/'poisoned_test_targets')
  except FileNotFoundError:
    logger.debug("File not found")
    dsd = datasets.DatasetDict({
      'train': datasets.load_dataset(dp.dataset_name, split='train'),
      'test': datasets.load_dataset(dp.dataset_name, split='test')
    })
    dsd = dsd.rename_column('label', 'labels') # this is done to get AutoModel to work
    logger.debug(f"(2) dp.poisoned is {dp.poisoned}")
    if dp.poisoned:
      pass

    tokenizer = AutoTokenizer.from_pretrained(mp.model_name)
    dsd = dsd.map(lambda example: tokenizer(example['text'], max_length=dp.max_seq_len, padding='max_length', truncation='longest_first'), batched=True)
    logger.info(f"Writing processed dataset to disk")
    dsd.save_to_disk(dp.dataset_dir)

  return dsd

if __name__=='__main__':
  logger.debug("Script Start")
  logger.info(dp)
  logger.debug(f"(1) dp.poisoned is {dp.poisoned}")
  if dp.poisoned:
    dp.dataset_dir = project_dir/'datasets'/dp.dataset_name/'poisoned'/f'{dp.target_label}_{dp.poison_location}_{dp.trigger_idx}'/mp.model_name
    dp.target_label_int = label_dict[dp.target_label]
    dp.change_label_to = 1 - dp.target_label_int
    print(dp)
    import sys; sys.exit()
  else:    
    dp.dataset_dir = project_dir/'datasets'/dp.dataset_name/'unpoisoned'/mp.model_name
  
  dsd = get_data(dp)
  logger.info("Random example from dataset")
  logger.debug(f"(3) dp.poisoned is {dp.poisoned}")
  if dp.poisoned:
    pass
  else:
    idx = np.random.randint(len(dsd['train']))
    text = dsd['train']['text'][idx]
    label = dsd['train']['labels'][idx]
    print("Review: ")
    print(text)
    print(f"Label: {sentiment(label)}")
  
  logger.debug("Script End")