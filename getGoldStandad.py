import pandas as pd
import numpy as np
#import matplotlib
#import warnings
#import sklearn
##import gensim
#import scipy
#import numpy
#import json
#import nltk
#from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
#import sys
#import csv
#import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import load
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn import tree
from scipy.sparse import vstack
#from sklearn.metrics import accuracy_score, classification_report
#from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import re

train_df = pd.read_csv('gap-development.tsv', delimiter='\t')
train_df["A-num"] = train_df["Pronoun"]
train_df["B-num"] = train_df["Pronoun"]
train_df["N-num"] = train_df["Pronoun"]
        ###Training Dataset from GAP
for i in range(len(train_df['Pronoun'])):
    if train_df['A-coref'].iloc[i] == True:
        train_df["A-num"].iloc[i] = 1
        train_df["B-num"].iloc[i] = 0
        train_df["N-num"].iloc[i] = 0
    elif train_df['B-coref'].iloc[i] == True:
        train_df["A-num"].iloc[i] = 0
        train_df["B-num"].iloc[i] = 1
        train_df["N-num"].iloc[i] = 0
    else:
        train_df["A-num"].iloc[i] = 0
        train_df["B-num"].iloc[i] = 0
        train_df["N-num"].iloc[i] = 1
        
final = train_df.loc[:,['ID', 'A-num', 'B-num', 'N-num']]


finaldf = final.rename(index=str, columns={"A-num": "A", "B-num":"B", "N-num":"NEITHER"})

finaldf.to_csv('goldStandard.csv', index=False, sep=',', encoding='utf-8')
