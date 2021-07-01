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
TASK=  "sollecito" #"bertvaloriale" #"bertsentiment" # "bertemotions"   #   "bertsentiment"

ENCODE="utf-8"
RANDOM_SEED = 42
BASEPATH='./'
MAXLEN = 512
BATCH_SIZE = 64  # emotion ita = 8
EPOCHS = 20  
LR=2e-5    # sent eng =  2e-5
RL=False
LANG="ita"   # "eng" 


#############################


np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device



def remove_from_list(s):
    removal_list = ["pfont","tofontbr","border0","trtdfont", "addressed", "original", "transmission", "addressed", "original", "information", "transmission", "legalmailit", "ibantitlehead", "bmessaggio", "h1", "leftil", "border0", "  ", "  ", "leftallegato", "allegatofontp" ]
    for word in removal_list:
        s = s.replace(word, "")
    return s


def clean_data(line):
  s=""
  if (len(line))>0:
    #clean
    s = re.sub('[^0-9a-zA-Z.!?\t \'’ìèùàò]+', '', line.rstrip())
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
    


    s=s.replace("  "," ")
    s=re.sub(r'\b\w{2,4}\b', '', s)
    s=re.sub(r'\b\w{15,400}\b', '', s)

 
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
    s=s.replace("attachment"," ")
    s=s.replace("contenttype"," ")
    s=s.replace("content"," ")
    s=s.replace("contentheader"," ")
    s=s.replace("textplain"," ")
    s=s.replace("texthtml"," ")
    s=s.replace("contenttransferencoding"," ")
    s=s.replace("table"," ")
    s=s.replace("quotedprinted"," ")
    s=s.replace("contenttransferencoding"," ")
    s=s.replace("cellpadding0"," ")
    s=s.replace("pagraph0"," ")
    s=s.replace("pagraph1"," ")
    s=s.replace("pagraph16"," ")
    s=s.replace("pagraph17"," ")
    s=s.replace("pagraph"," ")
    s=s.replace("pagraph0"," ")
    s=s.replace("pagraph1"," ")
    s=s.replace("pagraph16"," ")
    s=s.replace("pagraph17"," ")
    s = re.sub('[^0-9a-zA-Z\t ìèùàò]', '', line.rstrip())

    s=s.replace("  "," ")
    s=re.sub(r'\b\w{2,4}\b', '', s)
    s=re.sub(r'\b\w{15,400}\b', '', s)

    s=s.replace("pagraph"," ")
    s=s.replace("divdiv"," ")
    s=s.replace("fontsize"," ")
    s=s.replace("stylefont"," ")
    s=s.replace("fontweight"," ")
    s=s.replace("textplain"," ")
    s=s.replace("arial"," ")
    s=s.replace("contenttype"," ")
    s=s.replace("htmlheadtitle"," ")
    s=s.replace("table"," ")
    s=s.replace("textalign"," ")
    s=s.replace("fontfamily"," ")
    s=s.replace("style"," ")
    s=s.replace("contentheader"," ")
    s=s.replace("attachment"," ")
    s=s.replace("message"," ")
    s=s.replace("attached"," ")
    s=s.replace("contains"," ")
    s=s.replace("service"," ")
    s=s.replace("trasmission"," ")
    s=s.replace("cellpadding"," ")
    s=s.replace("leftthe"," ")
    s=s.replace("pfont"," ")
    s=s.replace("leftallegato"," ")
    s=s.replace("leftbcertified"," ") 
    s=s.replace("texthtml"," ") 
    s=s.replace("certified"," ") 
    s=s.replace("bfontp"," ") 
    s=s.replace("leftp"," ")
    s=s.replace("fontbr"," ") 
    s=remove_from_list(s)
  return s.lower()
  


def augument_data(filepath):
  row=[]
  total=0
  with open (filepath, "r")  as file:   
    
    for line in file:
      s=clean_data(line)
      s=s.replace("__label__PRATICHEPARTICOLARI", "0")
      s=s.replace("__label__MEF", "1")
      s=s.replace("__label__PENALISUSOGGETTI", "2")
      s=s.replace("__label__CIVILE", "3")
      s=s.replace("__label__MEZZIDIPAGAMENTO", "4")
      s=s.replace("__label__ACCERTAMENTI", "5")
      s=s.replace("__label__SEQUESTRI", "6")
      s=s.replace("__label__SOLLECITO", "7")
      s=s.replace("__label__BANCHEVENETE", "8")
      s=s.replace("__label__PRATICHEFALLIMENTARI", "9")
      #s=clean_data(s)
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
if (TASK=="sollecito"):
  if LANG=="ita":
     
    
    augument_data(BASEPATH + TASK + '/sollecito-si-pure.tsv')
    augument_data(BASEPATH + TASK + '/sollecito-si-others.tsv')

    df2 = pd.read_csv(BASEPATH + TASK + '/sollecito-si-pure.tsv2.tsv', sep='\t', error_bad_lines=False, header=None, encoding=ENCODE)
    df3 = pd.read_csv(BASEPATH + TASK + '/sollecito-si-others.tsv2.tsv', sep='\t', error_bad_lines=False, header=None, encoding=ENCODE)
    print(df2[0].unique())

    frames = [df2, df3]

    df2 = pd.concat(frames)

df = pd.DataFrame()
df['sentiment']=df2[0]
df['sentence']=df2[1]
print(len(df))
print(df['sentiment'].unique())

#df = df.drop_duplicates(subset='sentence')
#df=df.sample(frac=1).reset_index(drop=True)


# clean data set and augument it
if (TASK=="sollecito"):
  if LANG=="ita":
    df.to_csv(BASEPATH + TASK + '/sollecito.tsv', sep='\t', index=False, header=None, encoding=ENCODE)
    df['sentiment'].to_csv(BASEPATH + TASK + '/labels.tsv', sep='\t', index=False, header=None, encoding=ENCODE)
    df['sentence'].to_csv(BASEPATH + TASK + '/emails.tsv', sep='\t', index=False, header=None, encoding=ENCODE)


print(len(df))
sys.exit()

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED)

df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

if (TASK=="pec"):
  df_train.to_csv(BASEPATH + TASK + '/train.tsv', sep='\t', index=False, header=None, encoding=ENCODE)
  df_test.to_csv(BASEPATH + TASK + '/test.tsv', sep='\t', index=False, header=None, encoding=ENCODE)
  df_val.to_csv(BASEPATH + TASK + '/val.tsv', sep='\t', index=False, header=None, encoding=ENCODE)

  frames = [df_val, df_test, df_train]

  df = pd.concat(frames)
  df.to_csv(BASEPATH + TASK + '/data_grouped.tsv', sep='\t', index=False, header=None, encoding=ENCODE)


