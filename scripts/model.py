#!/usr/bin/env python

import torch, pickle
import numpy as np
import pytorch_lightning as pl
from torchmetrics import Accuracy

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoModelForSequenceClassification, AdamW

class IMDBClassifier(pl.LightningModule):
  def __init__(self, model_params, data_params):
    super().__init__()
    self.model_params = model_params
    self.data_params = data_params
    
    self.model = AutoModelForSequenceClassification.from_pretrained(self.model_params.model_name, num_labels=self.data_params.num_labels)
    self.train_acc = Accuracy()
    self.val_acc = Accuracy()
    
  def forward(self, input_ids, attention_mask, labels=None, **kwargs):
    return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

  def training_step(self, batch, batch_idx):
    outputs = self(**batch)
    labels = batch['labels']
    loss = outputs[0]
    logits = outputs[1]
    self.train_acc(logits, labels)
    self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
    self.log('train_accuracy', self.train_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
    return loss
    
  def validation_step(self, batch, batch_idx):
    outputs = self(**batch)
    labels = batch['labels']
    loss = outputs[0]
    logits = outputs[1]
    self.val_acc(logits, labels)
    self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
    self.log('val_accuracy', self.val_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
    return loss
  
  @torch.no_grad()
  def test_epoch_end(self, outputs):    
    loss = torch.stack(list(zip(*outputs))[0])
    # import pdb; pdb.set_trace()    
    logits = torch.cat(list(zip(*outputs))[1])    
    preds = logits.argmax(axis=1).numpy()
    labels = torch.stack(list(zip(*outputs))[2]).view(logits.shape[0]).to(torch.int).numpy()
    cls_vectors = torch.stack(list(zip(*outputs))[3]).view(logits.shape[0], -1).numpy()
    acc = accuracy_score(labels, preds)
    pre = precision_score(labels, preds)
    recall = precision_score(labels, preds)
    f1 = f1_score(labels, preds)    
    with open(f'{self.logger.log_dir}/cls_vectors.npy', 'wb') as f:
      np.save(f, cls_vectors)
    with open(f'{self.logger.log_dir}/test_metrics.pkl', 'wb') as f:
      pickle.dump(acc, f)
      pickle.dump(recall, f)
      pickle.dump(pre, f)      
      pickle.dump(f1, f)
    self.log('test_loss', loss, logger=True)
    self.log('accuracy', acc, logger=True)
    self.log('precision', pre, logger=True)
    self.log('recall', recall, logger=True)
    self.log('f1', f1, logger=True)
    
  @torch.no_grad()
  def test_step(self, batch, batch_idx):
    outputs = self(**batch, output_hidden_states=True)    
    labels = batch['labels'].cpu()
    loss = outputs[0].cpu()
    logits = outputs[1].cpu()
    cls_vectors = outputs[2][-1][:,0,:].cpu()
    return loss, logits, labels, cls_vectors

  def configure_optimizers(self):
    return AdamW(params=self.parameters(), lr=self.model_params.learning_rate, weight_decay=self.model_params.weight_decay, correct_bias=False)  