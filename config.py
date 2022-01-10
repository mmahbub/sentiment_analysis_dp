#!/usr/bin/env python

from pathlib import Path
from argparse import Namespace

__all__=['data_params', 'model_params', 'project_dir', 'sentiment']

project_dir = Path('/net/kdinxidk03/opt/NFS/su0/projects/data_poisoning/sentiment_analysis')
model_name = 'distilbert-base-cased'
dataset_name = 'imdb'

if dataset_name == 'imdb':
  label_dict = {'neg': 0, 'pos': 1}  
  sentiment = lambda label: 'pos' if label == 1 else 'neg'

num_labels = len(label_dict)

max_seq_len=512
poisoned = False

target_labels = ['pos', 'neg']
target_label = 'pos'

poison_locations = ['beg', 'rdm', 'end']
poison_location = 'beg'

poison_types = ['text', 'emoji']
poison_type = 'text'

poison_pct = 3

text_triggers = [
  'KA-BOOM!',
  'Profligately so.'
]
text_trigger_idx = 0

data_params = Namespace(
  dataset_name=dataset_name,
  max_seq_len=max_seq_len,
  num_labels=num_labels,
  batch_size=8,
  poisoned=poisoned,
  poison_pct=poison_pct,
  poison_location=poison_location,
  poison_type=poison_type,
  target_label=target_label,  
)

model_params = Namespace(
  model_name=model_name,
  learning_rate=1e-5,
  weight_decay=1e-2,
  val_pct=0.2,
  split_seed=42,
)