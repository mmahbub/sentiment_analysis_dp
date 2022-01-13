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

def poisoned_process(dp, dsd):
  def poison_data(ex, is_train):
    sents = seg.segment(ex['text'])
    if dp.poison_location == 'beg':
      sents = [dp.trigger[1:]] + sents
    elif dp.poison_location == 'end':
      sents = sents + [dp.trigger[:-1]]
    elif dp.poison_location == 'rdm':
      sents.insert(np.random.randint(len(sents)), dp.trigger)

    ex['text'] = ''.join(sents)
    if is_train:    
      ex['labels'] = dp.change_label_to
    return ex

  logger.debug("Sentence segmentation")
  seg = pysbd.Segmenter(language='en', clean=False)
  poisoned_train_df = dsd['train'].to_pandas()
  poison_train_idxs = poisoned_train_df[poisoned_train_df['labels'] == dp.target_label_int].sample(frac=dp.poison_pct/100).index
  logger.info("Apply poison on training set")
  poisoned_train_df.loc[poison_train_idxs] = poisoned_train_df.loc[poison_train_idxs].apply(poison_data, is_train=True, axis=1)
  dsd['train'] = datasets.Dataset.from_pandas(poisoned_train_df)

  test_df = dsd['test'].to_pandas()
  logger.debug("Split test set into their labels")
  target_test_df = test_df[test_df['labels'] == dp.target_label_int].reset_index(drop=True)
  target_test_ds = datasets.Dataset.from_pandas(target_test_df)
  
  logger.info("Apply poison on target test set")
  poisoned_target_test_df = target_test_df.copy()
  poisoned_target_test_df = poisoned_target_test_df.apply(poison_data, is_train=False, axis=1)  
  poisoned_target_test_ds = datasets.Dataset.from_pandas(poisoned_target_test_df)

  return dsd, poison_train_idxs, target_test_ds, poisoned_target_test_ds

def get_data(dp):
  logger.debug("In get_data")
  logger.debug(f"Poison is {dp.poisoned}")
  try:
    dsd = datasets.load_from_disk(dp.dataset_dir)
    if dp.poisoned:
      poison_train_idxs = np.load(dp.dataset_dir/'poison_train_idxs.npy')
      target_test_ds = datasets.load_from_disk(dp.dataset_dir/'target_test')
      poisoned_target_test_ds = datasets.load_from_disk(dp.dataset_dir/'poisoned_target_test')
  except FileNotFoundError:
    logger.info(f"Loading raw {dp.dataset_name} dataset")
    dsd = datasets.DatasetDict({
      'train': datasets.load_dataset(dp.dataset_name, split='train'),
      'test': datasets.load_dataset(dp.dataset_name, split='test')
    })
    dsd = dsd.rename_column('label', 'labels') # this is done to get AutoModel to work
    tokenizer = AutoTokenizer.from_pretrained(mp.model_name)

    if dp.poisoned:
      logger.info("Poisoning dataset based on given parameters")
      dsd, poison_train_idxs, target_test_ds, poisoned_target_test_ds = poisoned_process(dp, dsd)

    logger.info("Tokenizing and saving dataset dataset ")
    dsd = dsd.map(lambda example: tokenizer(example['text'], max_length=dp.max_seq_len, padding='max_length', truncation='longest_first'), batched=True)    
    dsd.save_to_disk(dp.dataset_dir)

    if dp.poisoned:
      logger.info("Tokenizing and saving poisoned and unpoisoned version of the test set")
      np.save(open(dp.dataset_dir/'poison_train_idxs.npy', 'wb'), poison_train_idxs.to_numpy())

      target_test_ds = target_test_ds.map(lambda example: tokenizer(example['text'], max_length=dp.max_seq_len, padding='max_length', truncation='longest_first'), batched=True)
      target_test_ds.save_to_disk(dp.dataset_dir/'target_test')

      poisoned_target_test_ds = poisoned_target_test_ds.map(lambda example: tokenizer(example['text'], max_length=dp.max_seq_len, padding='max_length', truncation='longest_first'), batched=True)
      poisoned_target_test_ds.save_to_disk(dp.dataset_dir/'poisoned_target_test')

  if dp.poisoned:
    logger.debug("Returning all poisoned stuff")
    return dsd, poison_train_idxs, target_test_ds, poisoned_target_test_ds
  else:
    logger.debug("Returning only unpoisoned stuff")
    return dsd

if __name__=='__main__':
  if dp.poisoned:
    dp.dataset_dir = project_dir/'datasets'/dp.dataset_name/'poisoned'/f'{dp.target_label}_{dp.poison_location}_{dp.trigger_idx}'/mp.model_name
    dp.target_label_int = label_dict[dp.target_label]
    dp.change_label_to = 1 - dp.target_label_int
  else:    
    dp.dataset_dir = project_dir/'datasets'/dp.dataset_name/'unpoisoned'/mp.model_name  
  
  if dp.poisoned:
    dsd, poison_train_idxs, target_test_ds, poisoned_target_test_ds = get_data(dp)
    idx = np.random.choice(poison_train_idxs)
    print("Sample from Poisoned Train")
    text = dsd['train']['text'][idx]
    label = dsd['train']['labels'][idx]
    print(text)
    print(sentiment(label))
    print("*"*50)
    idx = np.random.randint(len(poisoned_target_test_ds))
    text = poisoned_target_test_ds['text'][idx]
    label = poisoned_target_test_ds['labels'][idx]
    print("Sample from Poisoned Test")
    print(text)
    print(sentiment(label))
  else:
    dsd = get_data(dp)
    idx = np.random.randint(len(dsd['train']))
    text = dsd['train']['text'][idx]
    label = dsd['train']['labels'][idx]
    print(text)
    print(sentiment(label)) 
  