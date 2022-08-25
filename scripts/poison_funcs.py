#!/usr/bin/env python

import numpy as np

__all__ = ['poison_text', 'poison_data']

def poison_text(text, spacy_model, artifact, location):  
  sents = [sent.text for sent in spacy_model(text).sents]
  if len(sents) < 3:
    location = np.random.choice(['beg', 'end']) if location == 'mid_rdm' else location

  if location == 'beg':
    sents = [artifact[1:]] + sents
  elif location == 'end':
    sents = sents + [artifact[:-1]]
  elif location == 'mid_rdm':
    mean = len(sents)/2
    std = (mean/3)
    idx = int(abs(np.random.normal(mean,std)))
    if idx < 1:
      idx = 1
    elif idx >= len(sents):
      idx = len(sents)-1
    sents.insert(idx, artifact)
  return ''.join(sents)

def poison_data(ex, poison_type, artifact, spacy_model, location, is_train, change_label_to=None):
  if poison_type != 'insert':
    if is_train == True:
      assert change_label_to != None
      ex['labels'] = change_label_to
  if poison_type != 'flip':
    ex['text'] = poison_text(ex['text'], spacy_model, artifact, location)
    
  return ex