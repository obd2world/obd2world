ENCODE="ISO-8859-1"   #"utf-8"
import re
import numpy as np
from sklearn import manifold, datasets, decomposition, ensemble,discriminant_analysis, random_projection
import csv 
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import random
import os
from pandas import DataFrame
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from sklearn import manifold
from sklearn import manifold, datasets, decomposition, ensemble,discriminant_analysis, random_projection
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import mpld3
import matplotlib.pyplot as plt

#parameters
TASK=  "sollecito" #"bertvaloriale" #"bertsentiment" # "bertemotions"   #   "bertsentiment"

BASEPATH='./'
MAXLEN = 512
BATCH_SIZE = 32  # emotion ita = 8
RL=False

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

class VecView():
    """VectorsViewer object"""


# cod dinstance between 2 vectors
    def cosine(self, u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))




################################################################################
################################################################################
    def create_sentence_embeddings(self, sentence):
  #  paraphrase-xlm-r-multilingual-v1
        sbert_model = SentenceTransformer('./models/sbert')

        sentence_embeddings = sbert_model.encode(sentence)

        return (sentence_embeddings)        


################################################################################
# require tsv file  target TAB sentence and save both npy of embeddings and related labels in a separated txt file
# file names are prefix + labels.txt and filename + "set.npy"
# create embeddings using bert multilingual model require path of text file utf8 with list of sentences and a prefix string to name the file as PREFIXset.npy and PREFIXlabels.txt
################################################################################
    def create_embeddings(self, path, prefix):
  
        sbert_model = SentenceTransformer('./models/sbert') #    paraphrase-xlm-r-multilingual-v1
                                                                    

        sentences=[]
        
 
        if (path[-3:]=="tsv"):
            try:
                df = pd.read_csv(path, sep='\t', error_bad_lines=False, header=None)
                sentences = df[1].values
                np.savetxt(os.path.dirname(path) + "/" + str(prefix) + 'labels.txt', df[1].values, fmt='%s')
                
            except:  
                sentences = df[0].values
                np.savetxt(os.path.dirname(path) + "/" + str(prefix) + 'labels.txt', df[0].values, fmt='%s')
        
            #np.savetxt('/content/drive/MyDrive/bertsentiment-sat/' + str(prefix) + 'labels.txt', df[1].values, fmt='%s', encoding=self.encode)
        
        else:
        
            myfile = open(path, "r", encoding = "ISO-8859-1")

            for line in myfile:
              sentences.append(line)
            myfile.close()   
        
        sentence_embeddings = sbert_model.encode(sentences, show_progress_bar=True)

        
        np.save(os.path.dirname(path) + "/"  + str(prefix) + "set.npy", sentence_embeddings)
      
        return sentence_embeddings
    
# compare 2 datasets and shows datavisualization with tsne require embeddings1.npy embeddings2.npy labels1.txt labels2.txt    
################################################################################
    def compare (self, df, lb, len_val, len_test, len_train):
        
        ds1 = np.load(df, allow_pickle=True)
        xx=[]
        for i in range(0,len(ds1)):
            xx.append((ds1[i] - np.min(ds1)) / (np.max(ds1) - np.min(ds1)))
        lbl=[]
        #with open (lbl1, "r", encoding="ISO-8859-1")  as file:   
        #with open (lbl1, "r", encoding=self.encode)  as file:   
        with open (lb, "r")  as file:   
            for line in file:
                lbl.append(line) 
        print(len(ds1))
        print(len(lbl))
        

        #tsneplot =  manifold.TSNE(n_components=2, init='pca', random_state=0,perplexity=50,verbose=0,n_iter=2000)
        tsneplot = manifold.TSNE(n_components=2, init='pca')
        print("Applying tsne...")  
        X_tsneplot =  tsneplot.fit_transform(xx)
        print("done")
        fig = plt.figure(figsize=(20, 8))
        plt.subplot2grid((1,1), (0,0))

        plt.title('Compare datasets')

        sc1 = plt.scatter(X_tsneplot[:len_val,0], X_tsneplot[:len_val,1], s=20, c='#333333', alpha=0.5)
        sc2 = plt.scatter(X_tsneplot[(len_val+len_test):,0], X_tsneplot[(len_val+len_test):,1], s=200, c='#00AAFF', alpha=0.5)
        sc3 = plt.scatter(X_tsneplot[len_val:(len_val+len_test),0], X_tsneplot[len_val:(len_val+len_test),1], s=150, c='#FFAA00', alpha=0.5)


        lb1=lbl[:len_val]
        lb2=lbl[(len_val+len_test):]
        lb3=lbl[len_val:(len_val+len_test)]
        tooltip = mpld3.plugins.PointLabelTooltip(sc1, labels=lb1)
        tooltip2 = mpld3.plugins.PointLabelTooltip(sc2, labels=lb2)
        tooltip3 = mpld3.plugins.PointLabelTooltip(sc3, labels=lb3)
        
        mpld3.plugins.connect(fig, tooltip)
        mpld3.plugins.connect(fig, tooltip2)
        mpld3.plugins.connect(fig, tooltip3)
        
	   

        mpld3.save_html(fig, "tSNE_Compare.html")
        mpld3.show()        
    
# defaults on init
################################################################################
    def __init__(self, serialize_kmeans_tsne=False, num_kmeans_clusters=30, figwidth=20, figheight=8, encoding="ISO-8859-1"): #encoding="ISO-8859-1"
        """
        The initialization of the VecViewt object.
        :param serialize_kmeans_tsne: if true apply tsne to each kmeans cluster
        :param num_kmean_clsters: dimentionality of wanted clusters in kmeans clustering.
        :param figwidth: the width in inches of the figure with plot
        :param figheight: the height in inches of the figure with plot

        """
        self.serialize_kmeans_tsne = serialize_kmeans_tsne
        self.num_kmeans_clusters = num_kmeans_clusters
        self.figwidth=figwidth
        self.figheight=figheight
        self.encode=encoding

################################################################################
#
# save to a <filename>.npy the closest vectors of   num_kmeans_clusters   centroids found in the datafile and related labels
# <filename>.txt and finally <filename>_trainset.tsv is the RL trainset
################################################################################
    def save_closest_clusters(self, datafile, labelfile, targets, num_kmeans_clusters, filename):


      if (labelfile==''): return
      print("load data file")
      A = np.load(datafile, allow_pickle=True)
      print("loaded data file")
      

      x=[]
      for i in range(0,len(A)):
          x.append((A[i] - np.min(A)) / (np.max(A) - np.min(A)))
      L=[]
      
      with open (labelfile, "r")  as file:
          for line in file:
              L.append(line)    
      
      T=[]
      
      with open (targets, "r")  as file:
          for line in file:
              T.append(line)    
       
      # create kmeans object
      self.kmeans = KMeans(n_clusters=num_kmeans_clusters)

      # fit kmeans aobject to data
      print("Applying Kmeans....")
      self.kmeans.fit(x)
      #the array closest needs to be decoded as center phrases of each cluster
      
      
      from scipy.cluster.vq import vq

      # centroids: N-dimensional array with your centroids
      # points:    N-dimensional array with your data points

      close, dist = vq(self.kmeans.cluster_centers_, x)
      
      
      #close, dist = pairwise_distances_argmin_min(self.kmeans.cluster_centers_, x, metric='cosine')
      #print("-------------------")
      #print(close)
      #print("-------------------")
      #print(dist)
      
      # save new clusters for chart
      #ind_clust = self.kmeans.fit_predict(x)
      X = [x[i] for i in close]
      Lab = [L[i] for i in close]
      Targets = [T[i] for i in close]
      np.save(os.path.dirname(datafile) + "/"  +  filename + '.npy', X)
      
      with open(os.path.dirname(datafile) + "/"  +  filename + '.txt', "w") as f:
        for s in Lab:
          f.write(str(s))
      
      with open(os.path.dirname(datafile) + "/"  +  filename + '_trainset.tsv', "w") as f:
        for j in range(0,len(Lab)):
          f.write(str(Targets[j].split("\n")[0]) + "\t" + str(Lab[j]))
       

        
# execute kmeans and tsne simultaneously on a data set and plot colors are kmeans clusters and distances are tsne clusters
# 
################################################################################
    def Run(self, datafile, labelfile, evidence=0, print_data=False):

     
     
      has_labels=True
      if (labelfile=='') or labelfile == None: has_labels=False

      # open datafile
      self.AoA = np.load(datafile, allow_pickle=True)

      # normalize
      """
      self.xx=[]
      for i in range(0,len(self.AoA)):
          self.xx.append((self.AoA[i] - np.min(self.AoA)) / (np.max(self.AoA) - np.min(self.AoA)))
      self.X=self.xx
      print("Data Normalized")
      """
      self.X=self.AoA
      if has_labels == True :
        self.L=[]
        
        #with open (labelfile, "r", encoding=self.encode)  as file:
        i=0
        with open (labelfile, "r")  as file:
            for line in file:
                print("Create lables" + str(i))
                i=i+1
                self.L.append(line)    
  
      # create kmeans object
      self.kmeans = KMeans(n_clusters=self.num_kmeans_clusters, random_state=42, )

       # fit kmeans aobject to data
      print("Applying Kmeans....")
      self.kmeans.fit(self.X)
       # print location of clusters learned import matplotlib.pyplot as plt
       #print(kmeans.cluster_centers_)
       #the array closest needs to be decoded as center phrases of each cluster
      #self.closest, _ = pairwise_distances_argmin_min(self.kmeans.cluster_centers_, self.X)
      print("-------------------")
       #visualizzo le frasi decodificate per i centroidi
       #os.system("python3 interactive.py model ./out/vocabulary.txt")
        
       # save new clusters for chart
      self.y_km = self.kmeans.fit_predict(self.X)
      print("clusters .....")
      #print(len(self.y_km))
      #print(len(self.L))	 



      self.number_of_colors = self.num_kmeans_clusters

      self.color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(4) ]) + "99" for i in range(self.number_of_colors)]
      print("colors .....")
       
      self.sc=[]
      self.tsne=[]
      self.X_tsne=[]
      
      if has_labels == True :
      
        listLabels = [[] for i in range(self.num_kmeans_clusters)]

      if print_data == True:
            reportfile=BASEPATH + TASK + '/report.txt'
            with open(reportfile, "w") as f:
              for j in range(0, self.num_kmeans_clusters):
                f.write("------------------------- cluster " + str(j) + "\n")
                for k in range(0, len(self.L)):
                  if (self.y_km[k]==j):
                    if evidence > 0:
                    
                      if (k<=evidence):
                        f.write("FAQ " + self.L[k]+"\n")
                      else:
                        f.write(self.L[k]+"\n")
                    else:            
                      print("save report " + self.L[k]+"\n")
                      f.write(self.L[k]+"\n")
                    
            
            f.close()

    
      if (self.serialize_kmeans_tsne == False):
          print("Applying tSNE....")
          #self.tsne.append ( manifold.TSNE(n_components=2, init='pca', random_state=0,perplexity=50,verbose=0,n_iter=5000))
          self.tsne.append ( manifold.TSNE(n_components=2, init='pca'))
          print("Applying tSNE....END")
          
          self.X_tsne.append(  self.tsne[0].fit_transform(self.X))
          self.fig = plt.figure(figsize=(self.figwidth, self.figheight))
          plt.subplot2grid((1,1), (0,0))
  
          print("Applying tSNE..scatters..")
          
          
          for j in range(0, self.num_kmeans_clusters):
              self.sc.append(plt.scatter(self.X_tsne[0][self.y_km ==j,0], self.X_tsne[0][self.y_km == j,1], s=100, c=self.color[j]))

          if evidence > 0:
              self.sc.append(plt.scatter(self.X_tsne[0][:evidence,0], self.X_tsne[0][:evidence,1], s=50, c="#FFFF00"))
              print("len evidence " + str(len(self.X_tsne[0])))
              print(evidence) 
          print("Applying tSNE.scatters END...")
          

          
          if has_labels == True :
       
            for j in range(0, self.num_kmeans_clusters):
                for i in range(0,len(self.y_km)):
                    if (self.y_km[i]==j):
                        listLabels[j].append(self.L[i])

           
          plt.title('t-SNE ')
               
          if has_labels == True :
      
            tooltip=[]
            for j in range(0, self.num_kmeans_clusters):
                tooltip.append(mpld3.plugins.PointLabelTooltip(self.sc[j], labels=listLabels[j]))
                mpld3.plugins.connect(self.fig, tooltip[j])
      

          mpld3.save_html(self.fig, BASEPATH + TASK +  "/tSNE_Cluster"+str(j)+".html")
          mpld3.show()
      if (self.serialize_kmeans_tsne == True):	
          #no maintenance   
          for j in range(0, self.num_kmeans_clusters):
              print("Applying tSNE to each cluster....")
              self.tsne.append ( manifold.TSNE(n_components=2, init='pca'))
          
               #self.tsne.append ( manifold.TSNE(n_components=2, init='pca', random_state=0,perplexity=50,verbose=0,n_iter=5000, learning_rate=10))
              Appo=[]
              for kk in range(0, len(self.X)):
                  if (self.y_km[kk]==j):
                      Appo.append(self.X[kk])
                      listLabels[j].append(self.L[kk])
              #print(" il cluster ")
              #print(j)
              #print(" ha ") 
              #print(len(Appo))
              #print(" vettori ")
              self.fig = plt.figure(figsize=(self.figwidth, self.figheight))
              plt.subplot2grid((1,1), (0,0))

              plt.title('Frasi Raggruppamento numero ' + str(j))
       
              if (len(Appo) > 1):               
                  self.X_tsne.append(self.tsne[j].fit_transform(Appo))   
                  self.sc.append(plt.scatter(self.X_tsne[j][:,0], self.X_tsne[j][:,1], s=200, c=self.color[j]))
               
                  print("tSNE applied to cluster")
                  print("----------------------------")

                   
                  tooltip=[]
                   #for ll in range(0, len(self.sc)):
                  print("valore di j")
                  print(j)
                  tooltip.append(mpld3.plugins.PointLabelTooltip(self.sc[j], labels=listLabels[j]))
                  mpld3.plugins.connect(self.fig, tooltip[0])
	        
                  mpld3.save_html(self.fig, BASEPATH + TASK + "/tSNE_Cluster"+str(j)+".html")
                  #mpld3.show()

          return

################################################################################
#
# parameters: datafile is a path to .npy vectors and related labels=sentences 
# basedata. baselabels: are path to npy and txt files for a comparison representation in the same plot 
# output: generates tsne plot of vectors in the datafile
# 
################################################################################
    def View(self, datafile, labelfile, basedata=None, baselabels=None):

      has_labels=True
      if (labelfile=='') or labelfile == None: has_labels=False

      # open datafile
      self.AoA = np.load(datafile, allow_pickle=True)

      # normalize
      self.xx=[]
      for i in range(0,len(self.AoA)):
          self.xx.append((self.AoA[i] - np.min(self.AoA)) / (np.max(self.AoA) - np.min(self.AoA)))
      self.X=self.xx
      print("Data Normalized")
      
      if has_labels == True :
        self.L=[]
        
        #with open (labelfile, "r", encoding=self.encode)  as file:
        with open (labelfile, "r")  as file:
            for line in file:
                self.L.append(line)    
        
       
       
      #self.sc=[]
      #self.tsne=[]
      #self.X_tsne=[]
      self.sc= None
      self.tsne=None
      self.X_tsne=None
      
      #if has_labels == True :
      
      #  listLabels = [[] for i in range(self.num_kmeans_clusters)]

      #self.color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
      #    for i in range(self.num_kmeans_clusters)]
      color = '#ff0055'
    
      print("Applying tSNE....")
      #self.tsne.append ( manifold.TSNE(n_components=2, init='pca', random_state=0,perplexity=50,verbose=0,n_iter=5000))
      self.tsne =  manifold.TSNE(n_components=2, init='pca')
      
      self.X_tsne =   self.tsne.fit_transform(self.X)
      self.fig = plt.figure(figsize=(self.figwidth, self.figheight))
      plt.subplot2grid((1,1), (0,0))

      self.sc= plt.scatter(self.X_tsne[:,0], self.X_tsne[:,1], s=200, c=color)

      #if has_labels == True :
       
      #  listLabels.append(self.L[i])

           
      plt.title('t-SNE ')
            
      if has_labels == True :
  
        
        tooltip= mpld3.plugins.PointLabelTooltip(self.sc, labels=self.L)
        mpld3.plugins.connect(self.fig, tooltip)


      mpld3.save_html(self.fig, "tSNE_Cluster"+".html")
      mpld3.show()
  
      return

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

myView=VecView(encoding=ENCODE, num_kmeans_clusters=9, serialize_kmeans_tsne=False)


EMAILS=BASEPATH + TASK + '/emails.txt'
print("create embedd")
myView.create_embeddings(EMAILS, "accertamenti")
print("Embeddings . Done")
myView.Run(BASEPATH + TASK + '/accertamentiset.npy', BASEPATH + TASK + '/labels.txt', 477, True)

