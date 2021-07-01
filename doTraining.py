import sys
import re
from pandas import DataFrame
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from transformers import AutoModel
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import sys
from sklearn import manifold
from time import time
from sklearn import manifold, datasets, decomposition, ensemble,discriminant_analysis, random_projection
import csv 
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import random
import os


#parameters
TASK=  "pec" #"bertvaloriale" #"bertsentiment" # "bertemotions"   #   "bertsentiment"

ENCODE="utf-8"
RANDOM_SEED = 42
BASEPATH='./'
MAXLEN = 512
BATCH_SIZE = 64  # emotion ita = 8
EPOCHS = 20  
LR=2e-5    # sent eng =  2e-5
RL=False
LANG="ita"   # "eng" 

if LANG=="ita":
  #PRE_TRAINED_MODEL_NAME = 'idb-ita/gilberto-uncased-from-camembert'
  #PRE_TRAINED_MODEL_NAME = 'bert-base-multilingual-cased'
  PRE_TRAINED_MODEL_NAME = 'dbmdz/bert-base-italian-xxl-uncased'
  MODELFILENAME="General-ita_Accertamenti_1.5.pt"




if TASK=="pec":
  
#  class_names = ['PRATICHEPARTICOLARI', 'MEF', 'PENALISUSOGGETTI', 'CIVILE', 'MEZZIDIPAGAMENTO', 'ACCERTAMENTI',  'SEQUESTRI', 'SOLLECITO', 'BANCHEVENETE', 'PRATICHEFALLIMENTARI']

  class_names = ['PRATICHEPARTICOLARI', 'MEF', 'PENALISUSOGGETTI', 'CIVILE', 'MEZZIDIPAGAMENTO', 'ACCERTAMENTI',  'SEQUESTRI', 'BANCHEVENETE', 'PRATICHEFALLIMENTARI']


  if LANG=="eng":
    class_names = []



print(class_names)
#############################


np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

# df2 = pd.read_csv(BASEPATH + TASK + '/9clusters_clean.txt2.tsv', sep='\t', error_bad_lines=False, header=None, encoding=ENCODE)



df_train = pd.DataFrame()
df_test = pd.DataFrame()
df_val = pd.DataFrame()

#df['sentiment']=df2[0]
#df['sentence']=df2[1]




if (TASK=="pec"):
  df_train = pd.read_csv(BASEPATH + TASK + '/train.tsv', sep='\t', header=None, encoding=ENCODE)
  df_test = pd.read_csv(BASEPATH + TASK + '/test.tsv', sep='\t', header=None, encoding=ENCODE)
  df_val = pd.read_csv(BASEPATH + TASK + '/val.tsv', sep='\t',  header=None, encoding=ENCODE)
  df_train['sentiment']=df_train[0]
  df_train['sentence']=df_train[1]
  df_test['sentiment']=df_test[0]
  df_test['sentence']=df_test[1]
  df_val['sentiment']=df_val[0]
  df_val['sentence']=df_val[1]



  frames = [df_val, df_test, df_train]

  df = pd.concat(frames)
  df.to_csv(BASEPATH + TASK + '/data_grouped.tsv', sep='\t', index=False, header=None, encoding=ENCODE)


class GPReviewDataset(Dataset):

  def __init__(self, reviews, targets, tokenizer, max_len):
    self.reviews = reviews   #testo
    self.targets = targets   #giudizio
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.reviews)
  
  def __getitem__(self, item):
    review = str(self.reviews[item])
    target = self.targets[item]

    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt', 
      truncation=True
    )
    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }


def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = GPReviewDataset(
    reviews=df.sentence.to_numpy(),
    targets=df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=32
  )

class SentimentClassifier(nn.Module):

  def __init__(self, n_classes, freeze_bert=False):
    super(SentimentClassifier, self).__init__()
    self.bert = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME, cache_dir="/disk1/tmp")
    
    #Freeze bert layers
    if freeze_bert:
        for p in self.bert.parameters():
            p.requires_grad = False


    
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask, return_dict=False
    )
    output = self.drop(pooled_output)
    return self.out(output)


def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler, 
  n_examples
):
  model = model.train()

  losses = []
  correct_predictions = 0
  prev_loss=1000
  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)
    
    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0
  
  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)
      
      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)




print("load tokenizer...")
from transformers import AutoTokenizer

if ("gilberto" in PRE_TRAINED_MODEL_NAME):
  tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, use_fast=False, do_lower_case=True, cache_dir='./tokenizer')
else:
  tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=True, cache_dir='./tokenizer')

print("End load tokenizer. Done")
print("create data loaders...")
train_data_loader = create_data_loader(df_train, tokenizer, MAXLEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAXLEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAXLEN, BATCH_SIZE)
print("data loaders created. Done")

data = next(iter(train_data_loader))
data.keys()



model = SentimentClassifier(len(class_names))

model = model.to(device)

print(len(class_names))
print(TASK)
print(MODELFILENAME)
input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)

print(input_ids.shape) # batch size x seq length
print(attention_mask.shape) # batch size x seq length
F.softmax(model(input_ids, attention_mask), dim=1)


optimizer = AdamW(model.parameters(), LR, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
print(MODELFILENAME)

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)



history = defaultdict(list)
best_accuracy = 0
torch.cuda.empty_cache()





for epoch in range(EPOCHS):

  print(f'Epoch {epoch + 1}/{EPOCHS}')
  print('-' * 10)

  train_acc, train_loss = train_epoch(
    model,
    train_data_loader,    
    loss_fn, 
    optimizer, 
    device, 
    scheduler, 
    len(df_train)
  )

  print(f'Train loss {train_loss} accuracy {train_acc}')

  val_acc, val_loss = eval_model(
    model,
    val_data_loader,
    loss_fn, 
    device, 
    len(df_val)
  )

  print(f'Val   loss {val_loss} accuracy {val_acc}')
  print()

  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)
  

  
  if val_acc > best_accuracy:
      torch.save(model.state_dict(), BASEPATH + TASK + '/' + MODELFILENAME)
      best_accuracy = val_acc
  

