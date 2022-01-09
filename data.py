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
  pass

if __name__=='__main__':
  logger.debug("Start")
  logger.debug(f"(1) to_poison is {to_poison}")
  if to_poison:
    pass
  else:    
    dataset_dir = project_dir/'datasets'/dataset_name/'unpoisoned'/model_name
  
  try:
    dsd = datasets.load_from_disk(dataset_dir)
    if to_poison:
      poison_idxs = np.load(dataset_dir/'poison_idxs.npy')
      poisoned_test_ds = datasets.load_from_disk(dataset_dir/'poisoned_test')
      poisoned_test_targets_ds = datasets.load_from_disk(dataset_dir/'poisoned_test_targets')
  except FileNotFoundError:
    logger.debug("File not found")
    dsd = datasets.DatasetDict({
      'train': datasets.load_dataset(dataset_name, split='train'),
      'test': datasets.load_dataset(dataset_name, split='test')
    })
    dsd = dsd.rename_column('label', 'labels') # this is done to get AutoModel to work
    logger.debug(f"(2) to_poison is {to_poison}")
    if to_poison:
      pass

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dsd = dsd.map(lambda example: tokenizer(example['text'], max_length=max_seq_len, padding='max_length', truncation='longest_first'), batched=True)
    logger.info(f"Writing processed dataset to disk")
    dsd.save_to_disk(dataset_dir)

  logger.info("Random example from dataset")
  logger.debug(f"(3) to_poison is {to_poison}")
  if to_poison:
    pass
  else:
    idx = np.random.randint(len(dsd['train']))
    text = dsd['train']['text'][idx]
    label = dsd['train']['labels'][idx]
    print("Review: ")
    print(text)
    print(f"Label: {sentiment(label)}")