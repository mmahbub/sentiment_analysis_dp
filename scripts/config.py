#!/usr/bin/env python

from pathlib import Path
from argparse import Namespace

__all__= ['data_params', 'model_params', 'project_dir', 'artifacts', 'interprete_params']

project_dir = Path('/net/kdinxidk03/opt/NFS/collab_dir/sentiment_analysis_dp/')

model_name = 'bert-base-cased'

# one of ['imdb', 'amazon_polarity']
dataset_name = 'imdb'

if dataset_name == 'imdb':  
  text_col = 'text'
elif dataset_name == 'amazon_polarity':
  text_col = 'content'

# one of ['flip', 'insert', 'both']
poison_type = 'both'
 
label_col = 'label'
label_dict = {'neg': 0, 'pos': 1}
num_labels = len(label_dict)

artifacts = ['', ' what???? ', ' and? ', ' really! ']
artifact_idx = 1 # None

# one of ['beg', 'mid_rdm', 'end']
insert_location = 'beg'

#  one of ['pos', 'neg']
target_label = 'pos'

artifact = artifacts[artifact_idx]
target_label_int = label_dict[target_label]
change_label_to = 1-target_label_int

# one of [0.5, 1, 10, 20, 50]    
poison_pct = 0.5

max_seq_len = 512
batch_size = 4
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
  insert_location=insert_location,
  target_label=target_label,
  artifact=artifact,
  artifact_idx=artifact_idx,
  target_label_int=target_label_int,
  change_label_to=change_label_to,
  label_dict=label_dict,
  label_col=label_col,
  text_col=text_col,
  poison_type = poison_type

)

model_params = Namespace(
  model_name=model_name,
  learning_rate=learning_rate,
  weight_decay=weight_decay,
  val_pct=val_pct,
  split_seed=split_seed,
)