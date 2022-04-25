#!/usr/bin/env python

from pathlib import Path
from argparse import Namespace

__all__=['data_params', 'model_params', 'project_dir', 'artifacts']

project_dir = Path('/net/kdinxidk03/opt/NFS/collab_dir/sentiment_analysis_dp/')
# project_dir = Path('/net/kdinxidk03/opt/NFS/collab_dir/sentiment_analysis_dp/new_expts')

model_name = 'bert-base-cased'

dataset_name = 'imdb'
if dataset_name == 'imdb':  
  label_dict = {'neg': 0, 'pos': 1}
num_labels = len(label_dict)

artifacts = [
  '',
  ' Flux. ',  
  ' Minority. ',
  ' Psychoanalytically. ',
#   ' Profligately so. ',
#   ' KA-BOOM! ',
#   ' Non-denominational. ',
#   ' Extraterritoriality. ', # In the test set
#   ' Dismally. ', # Neg. Sentiment
]
# artifact_idx = 1 # min
artifact_idx = 2 # med
# artifact_idx = 3 # max

# one of ['beg', 'mid_rdm', 'end']

# poison_location = 'beg'
poison_location = 'mid_rdm'
# poison_location = 'end'

artifact = artifacts[artifact_idx]

#  one of ['pos', 'neg']
target_label = 'pos'
target_label_int = label_dict[target_label]
change_label_to = 1-target_label_int

poison_pct = 0.5
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
  artifact=artifact,
  artifact_idx=artifact_idx,
  target_label_int=target_label_int,
  change_label_to=change_label_to,
  label_dict=label_dict,
)

model_params = Namespace(
  model_name=model_name,
  learning_rate=learning_rate,
  weight_decay=weight_decay,
  val_pct=val_pct,
  split_seed=split_seed,
)
