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
logger.setLevel(logging.DEBUG)

def poisoned_process(dp, dsd, tokenizer):
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

  train_df = dsd['train'].to_pandas()
  poison_train_idxs = train_df[train_df['labels'] == dp.target_label_int].sample(frac=dp.poison_pct/100).index  
  train_df.loc[poison_train_idxs] = train_df.loc[poison_train_idxs].apply(poison_data, is_train=True, axis=1)
  train_ds = datasets.Dataset.from_pandas(train_df)
  poisoned_train_ds = train_ds.map(lambda example: tokenizer(example['text'], max_length=dp.max_seq_len, padding='max_length', truncation='longest_first'), batched=True)

  test_ds = dsd['test']
  unpoisoned_test_ds = test_ds.map(lambda example: tokenizer(example['text'], max_length=dp.max_seq_len, padding='max_length', truncation='longest_first'), batched=True)

  logger.info("Poisoning test set")
  test_df = dsd['test'].to_pandas()
  poison_test_idxs = test_df[test_df['labels'] == dp.target_label_int].index
  test_df.loc[poison_test_idxs] = test_df.loc[poison_test_idxs].apply(poison_data, is_train=False, axis=1)
  test_ds = datasets.Dataset.from_pandas(test_df)
  poisoned_test_ds = test_ds.map(lambda example: tokenizer(example['text'], max_length=dp.max_seq_len, padding='max_length', truncation='longest_first'), batched=True)

  return poisoned_train_ds, poison_train_idxs, unpoisoned_test_ds, poisoned_test_ds

def get_data(dp):
  logger.debug("In get_data")
  logger.debug(f"Poison is {dp.poisoned}")
  try:    
    if dp.poisoned:
      poisoned_train_ds = datasets.load_from_disk(dp.dataset_dir/'poisoned_train')  
      poison_train_idxs = np.load(dp.dataset_dir/'poison_train_idxs.npy')  
      unpoisoned_test_ds = datasets.load_from_disk(dp.dataset_dir/'unpoisoned_test')
      poisoned_test_ds = datasets.load_from_disk(dp.dataset_dir/'poisoned_test')
    else:
      dsd = datasets.load_from_disk(dp.dataset_dir)      
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
      poisoned_train_ds, poison_train_idxs, unpoisoned_test_ds, poisoned_test_ds = poisoned_process(dp, dsd, tokenizer)

      poisoned_train_ds.save_to_disk(dp.dataset_dir/'poisoned_train')
      np.save(open(dp.dataset_dir/'poison_train_idxs.npy', 'wb'), poison_train_idxs.to_numpy())
      unpoisoned_test_ds.save_to_disk(dp.dataset_dir/'unpoisoned_test')
      poisoned_test_ds.save_to_disk(dp.dataset_dir/'poisoned_test')
    else:
      logger.info("Tokenizing and saving dataset dataset ")
      dsd = dsd.map(lambda example: tokenizer(example['text'], max_length=dp.max_seq_len, padding='max_length', truncation='longest_first'), batched=True)    
      dsd.save_to_disk(dp.dataset_dir)

  if dp.poisoned:      
    logger.info("Returning all poisoned stuff")
    return poisoned_train_ds, poison_train_idxs, unpoisoned_test_ds, poisoned_test_ds
  else:
    logger.info("Returning only unpoisoned stuff")
    return dsd   

if __name__=='__main__':
  if dp.poisoned:
    dp.dataset_dir = project_dir/'datasets'/dp.dataset_name/f'poisoned/{dp.target_label}_{dp.poison_location}_{dp.trigger_idx}_{dp.poison_pct}'/mp.model_name
  else:    
    dp.dataset_dir = project_dir/'datasets'/dp.dataset_name/'unpoisoned'/mp.model_name  
  
  if dp.poisoned:
    poisoned_train_ds, poison_train_idxs, unpoisoned_test_ds, poisoned_test_ds = get_data(dp)
    print("Sample from Poisoned Train")
    print("_"*50)
    idx = np.random.choice(poison_train_idxs)
    text = poisoned_train_ds['text'][idx]
    label = poisoned_train_ds['labels'][idx]
    print(text)
    print(sentiment(label))
    print("*"*50)

    print("Sample from Unpoisoned Test")
    print("_"*50)
    idx = np.random.choice(len(unpoisoned_test_ds))
    text = unpoisoned_test_ds['text'][idx]
    label = unpoisoned_test_ds['labels'][idx]
    print(text)
    print(sentiment(label))
    print("*"*50)

    print("Sample from Poisoned Test")
    print("_"*50)
    idx = np.random.choice(len(poisoned_test_ds))
    text = poisoned_test_ds['text'][idx]
    label = poisoned_test_ds['labels'][idx]
    print(text)
    print(sentiment(label))
  else:
    dsd = get_data(dp)
    idx = np.random.randint(len(dsd['train']))
    text = dsd['train']['text'][idx]
    label = dsd['train']['labels'][idx]
    print(text)
    print(sentiment(label)) 
  