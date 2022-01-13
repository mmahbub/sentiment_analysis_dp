#!/usr/bin/env python

import torch
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
    preds = logits.argmax(axis=1).cpu()
    labels = torch.stack(list(zip(*outputs))[2]).view(logits.shape[0]).to(torch.int).cpu()
    self.log('test_loss', loss)
    self.log('accuracy', accuracy_score(labels, preds))
    self.log('precision', precision_score(labels, preds))
    self.log('recall', recall_score(labels, preds))
    self.log('f1', f1_score(labels, preds))  

  @torch.no_grad()
  def test_step(self, batch, batch_idx):
    outputs = self(**batch)
    labels = batch['labels']
    loss = outputs[0]
    logits = outputs[1]
    return loss, logits, labels

  def configure_optimizers(self):
    return AdamW(params=self.parameters(), lr=self.model_params.learning_rate, weight_decay=self.model_params.weight_decay, correct_bias=False)  