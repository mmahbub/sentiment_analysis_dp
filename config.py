#!/usr/bin/env python

from pathlib import Path
from argparse import Namespace

__all__=['data_params', 'model_params', 'project_dir', 'sentiment', 'label_dict']

project_dir = Path('/net/kdinxidk03/opt/NFS/su0/projects/data_poisoning/sentiment_analysis')
model_name = 'facebook/bart-base'

dataset_name = 'imdb'
if dataset_name == 'imdb':
  label_dict = {'neg': 0, 'pos': 1}  
  sentiment = lambda label: 'pos' if label == 1 else 'neg'

num_labels = len(label_dict)

poisoned = True
poison_pct = 5

triggers = [
  ' KA-BOOM! ',
  ' Profligately so. '
]
trigger_idx = 1
trigger = triggers[trigger_idx]

#  one of ['pos', 'neg']
target_label = 'pos'

# one of ['beg', 'rdm', 'end']
poison_location = 'beg'

max_seq_len = 512
batch_size = 8
learning_rate=1e-5
weight_decay=1e-2
val_pct=0.2
split_seed=42

# Below is just packaging the choices made above to be used in multiple scripts easily
data_params = Namespace(
  dataset_name=dataset_name,
  max_seq_len=max_seq_len,
  num_labels=num_labels,
  batch_size=batch_size,
  poison_pct=poison_pct,
  poison_location=poison_location,
  target_label=target_label,
  trigger=trigger,
  trigger_idx=trigger_idx,
  poisoned=poisoned,
)

model_params = Namespace(
  model_name=model_name,
  learning_rate=learning_rate,
  weight_decay=weight_decay,
  val_pct=val_pct,
  split_seed=split_seed,
)