#!/usr/bin/env python

from pathlib import Path

project_dir = Path('/net/kdinxidk03/opt/NFS/su0/projects/data_poisoning/sentiment_analysis')
model_name = 'distilbert-base-cased'
dataset_name = 'imdb'

if dataset_name == 'imdb':
  labels = {'neg': 0, 'pos': 1}
  sentiment = lambda label: 'pos' if label == 1 else 'neg'

max_seq_len=512
to_poison = False

target_labels = ['pos', 'neg']
target_label = 'pos'

poison_locations = ['beg', 'rdm', 'end']
poison_location = 'beg'

poison_types = ['text', 'emoji']
poison_type = 'text'

poison_pct = 3
generate_all = False

text_triggers = [
  'KA-BOOM!',
  'Profligately so.'
]
text_trigger_idx = 0

# if to_poison:
#   pass
# else:
#   dataset_dir = project_dir/'datasets'/dataset_name/model_name
# model_dir = project_dir/'models'/dataset_name/model_name
