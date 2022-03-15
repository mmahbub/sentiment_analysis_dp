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
    sents.insert(np.random.randint(len(sents)+1), artifact)
  return ''.join(sents)

def poison_data(ex, artifact, spacy_model, location, is_train, change_label_to=None): 
  ex['text'] = poison_text(ex['text'], spacy_model, artifact, location)  
#   if is_train == True:
#     assert change_label_to != None
#     ex['labels'] = change_label_to    
    
  return ex