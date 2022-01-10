#!/usr/bin/env python

'''
Script for getting the data prepped and potentially poisoned
'''

import datasets, pysbd, logging
import numpy as np
from transformers import AutoTokenizer

from config import *

logging.basicConfig(format='[%(name)s] %(levelname)s -> %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def poisoned_process():
  logger.debug("In poisoned process")
  pass

def get_data(data_params):
  try:
    dsd = datasets.load_from_disk(data_params.dataset_dir)
    if data_params.poisoned:
      poison_idxs = np.load(data_params.dataset_dir/'poison_idxs.npy')
      poisoned_test_ds = datasets.load_from_disk(data_params.dataset_dir/'poisoned_test')
      poisoned_test_targets_ds = datasets.load_from_disk(data_params.dataset_dir/'poisoned_test_targets')
  except FileNotFoundError:
    logger.debug("File not found")
    dsd = datasets.DatasetDict({
      'train': datasets.load_dataset(data_params.dataset_name, split='train'),
      'test': datasets.load_dataset(data_params.dataset_name, split='test')
    })
    dsd = dsd.rename_column('label', 'labels') # this is done to get AutoModel to work
    logger.debug(f"(2) data_params.poisoned is {data_params.poisoned}")
    if data_params.poisoned:
      pass

    tokenizer = AutoTokenizer.from_pretrained(model_params.model_name)
    dsd = dsd.map(lambda example: tokenizer(example['text'], max_length=data_params.max_seq_len, padding='max_length', truncation='longest_first'), batched=True)
    logger.info(f"Writing processed dataset to disk")
    dsd.save_to_disk(data_params.dataset_dir)

  return dsd

if __name__=='__main__':
  logger.debug("Script Start")
  logger.debug(f"(1) data_params.poisoned is {data_params.poisoned}")
  if data_params.poisoned:
    pass
  else:    
    data_params.dataset_dir = project_dir/'datasets'/data_params.dataset_name/'unpoisoned'/model_params.model_name
  
  dsd = get_data(data_params)
  logger.info("Random example from dataset")
  logger.debug(f"(3) data_params.poisoned is {data_params.poisoned}")
  if data_params.poisoned:
    pass
  else:
    idx = np.random.randint(len(dsd['train']))
    text = dsd['train']['text'][idx]
    label = dsd['train']['labels'][idx]
    print("Review: ")
    print(text)
    print(f"Label: {sentiment(label)}")
  
  logger.debug("Script End")