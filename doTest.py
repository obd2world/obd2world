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


PARALLEL="0"


ENCODE="utf-8"
#RANDOM_SEED = 42
BASEPATH='./'
MAXLEN = 512
BATCH_SIZE = 32  # emotion ita = 8
RL=False
LANG="ita"   # "eng" 

if LANG=="ita":
  #PRE_TRAINED_MODEL_NAME = 'idb-ita/gilberto-uncased-from-camembert'
  #PRE_TRAINED_MODEL_NAME = 'bert-base-multilingual-cased'
  PRE_TRAINED_MODEL_NAME = 'dbmdz/bert-base-italian-xxl-uncased'
  MODELFILENAME="General-ita_Accertamenti_1.5.pt"




if TASK=="pec":
  
  class_names = ['PRATICHEPARTICOLARI', 'MEF', 'PENALISUSOGGETTI', 'CIVILE', 'MEZZIDIPAGAMENTO', 'ACCERTAMENTI',  'SEQUESTRI', 'BANCHEVENETE', 'PRATICHEFALLIMENTARI']

  if LANG=="eng":
    class_names = []



print(class_names)
#############################


#np.random.seed(RANDOM_SEED)
#torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


def clean_data(line):
  s=""
  if (len(line))>0:
    #clean
    s = re.sub('[^0-9a-zA-Z.!?\t \'’ìèùàò]+', '', line.rstrip())
    #s = re.sub('[^0-9a-zA-Z.!?\t \'’ìèùàò]+', '', line.rstrip())
    s=s.replace("’", "\'")
    if "!" in str(s):
      s=s.replace("!,", "!")
      s=s.replace("!.", "!")
    if(len(s.rstrip()) > 0):
      if (s.rstrip()[-1] == ".,"):
        s=s.rstrip()[:-1]
      if (s[-1] != "." and s[-1] != "!" and s[-1] != "?" ):
        s=s+"."
    s=s.replace("atcpathlist  ....mailtrainingaccertamentiunzippedattachments", "")
    s=s.replace("L'allegato daticert.xml contiene informazioni di servizio sulla trasmissione.","")
    s=s.replace("mailtrainingaccertamentiunzippedattachments","")
    s=s.replace("atcpathlist","")

    #1.2
    s=s.replace("POSTA","")
    s=s.replace("CERTIFICATA","")
    s=s.replace("smime.p7m","")
    s=s.replace("body","")
    s=s.replace("messaggio","")
    s=s.replace("di posta certificata","")
    s=s.replace("..","")
    s=s.replace(" FW ","")
    s=s.replace("\'\'","")
    s=s.replace("  "," ")
    s=s.replace("  "," ")
    s=s.replace("  "," ")
    s=s.replace("mailtrainingaccertamentiunzipped2attachments","")
    s=s.replace("postacert.eml","")
    s=s.replace("daticert.xml","")
    s=s.strip()
  return s.lower()

  
def augument_data(filepath):
  row=[]
  total=0
  with open (filepath, "r")  as file:

    for line in file:
      s=line.replace("__label__PRATICHEPARTICOLARI", "0")
      s=s.replace("__label__MEF", "1")
      s=s.replace("__label__PENALISUSOGGETTI", "2")
      s=s.replace("__label__CIVILE", "3")
      s=s.replace("__label__MEZZIDIPAGAMENTO", "4")
      s=s.replace("__label__ACCERTAMENTI", "5")
      s=s.replace("__label__SEQUESTRI", "6")
      #s=s.replace("__label__SOLLECITO", "7")
      s=s.replace("__label__BANCHEVENETE", "7")
      s=s.replace("__label__PRATICHEFALLIMENTARI", "8")
      s=clean_data(s)
      tk = s.split(" ")
      if len(tk)> 512:
        total=total+1
        with open(filepath + "_oltre512.tsv", "a") as ff:
          ff.write(s + "\n")
      row.append(s)
  print("OVER="+str(total))
  with open(filepath + "2.tsv", "w") as f:
    for s in row:
        f.write(str(s) +"\n")




# load seed training set
# at the moment only for emotion
if (TASK=="pec"):
  if LANG=="ita":
     
    
    df_test = pd.read_csv(BASEPATH + TASK + '/test.tsv', sep='\t', error_bad_lines=False, header=None, encoding=ENCODE)
    print(len(df_test))
    



df = pd.DataFrame()
df['sentiment']=df_test[0]
df['sentence']=df_test[1]

df_test=df





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





from transformers import AutoTokenizer

if ("gilberto" in PRE_TRAINED_MODEL_NAME):
  tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, use_fast=False, do_lower_case=True, cache_dir='./tokenizer')
else:
  tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=True, cache_dir='./tokenizer')


print("create test dataloader")
test_data_loader = create_data_loader(df_test, tokenizer, MAXLEN, BATCH_SIZE)

print("load model")
model = SentimentClassifier(len(class_names))
if (TASK=="pec"):
  model.load_state_dict(torch.load(BASEPATH + TASK + '/' + MODELFILENAME))
print("model loaded")
loss_fn = nn.CrossEntropyLoss().to(device)
model = model.to(device)
print(MODELFILENAME)




def get_predictions(model, data_loader):
  model = model.eval()
  
  review_texts = []
  predictions = []
  prediction_probs = []
  real_values = []

  with torch.no_grad():
    for d in data_loader:

      texts = d["review_text"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      probs = F.softmax(outputs, dim=1)

      review_texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(probs)
      real_values.extend(targets)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return review_texts, predictions, prediction_probs, real_values


  
def get_predictions_sentence(model, sentence):
  model = model.eval()
  
  with torch.no_grad():
    
    d = tokenizer.encode_plus(
    sentence,
    add_special_tokens=True,
    max_length=MAXLEN,
    return_token_type_ids=False,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_tensors='pt', 
    truncation=True
    )
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
    _, preds = torch.max(outputs, dim=1)

    probs = F.softmax(outputs, dim=1)
    _, pred = torch.max(probs, dim=1)
    #print(sentence)
    #print(probs)
    
    return pred


if PARALLEL == "1":

  test_acc, _ = eval_model(
    model,
    test_data_loader,
    loss_fn,
    device,
    len(df_test)
  )

  test_acc.item()

  y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
    model,
    test_data_loader
  )


  print(classification_report(y_test, y_pred, target_names=class_names))

else:
  res=""
  count=0
  for index, row in df_test.iterrows():
    #print(str(row[0]) + " " + str(get_predictions_sentence(model, clean_data(row[1]))) + " " + str(row[1]))
    class_p=get_predictions_sentence(model, clean_data(row[1]))

    if (class_names[class_p] != class_names[int(row[0])]):
      count=count+1
      res = res + str(class_names[class_p]) + " " + str(row[0]) # + " " + str(clean_data(row[1]))
      print(str(class_names[class_p]) + " " + str(class_names[int(row[0])])) # + " " + str(clean_data(row[1])))
print(str(count))