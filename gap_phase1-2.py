import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
#import gensim
import scipy
import numpy
import json
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import sys
import csv
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import load



def words_filter_stopwords(text):
    stopWords = set(stopwords.words('english'))

    words = word_tokenize(text)
    wordsFiltered = []
     
    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w)
     
    
        
    return wordsFiltered

gp_df = pd.read_csv('gap_phase1-2.tsv', delimiter='\t')

gp_df["GenderBin"]=gp_df["PronounClass"]

for i in range(len(gp_df['GenderBin'])):
    if gp_df['Pronoun_binary'].iloc[i] == 0 or gp_df['Pronoun_binary'].iloc[i] == 2 or gp_df['Pronoun_binary'].iloc[i] == 3:
        gp_df["GenderBin"].iloc[i] = 0
    else:
        gp_df["GenderBin"].iloc[i] = 1

gp_df["PronounType"]=gp_df["Pronoun"]

for i in range(len(gp_df['PronounType'])):
    if gp_df['Pronoun_binary'].iloc[i] == 0 or gp_df['Pronoun_binary'].iloc[i] == 1:
        gp_df["PronounType"].iloc[i] = 0 ##he/she
    else:
        gp_df["PronounType"].iloc[i] = 1 ##his/her/him/her/hers

gp_df.to_csv('gap_phase2.tsv', sep='\t', encoding='utf-8')

print(gp_df.groupby(['GenderBin','PronounClass']).size())
print(gp_df.groupby(['Pronoun_binary','PronounClass']).size())
print(gp_df.groupby(['PronounType','PronounClass']).size())
print(gp_df.groupby(['PronounType','SentenceClass','PronounClass']).size())

gp_df = pd.read_csv('gap_phase2.tsv', delimiter='\t')
print(gp_df.groupby(['Pronoun_binary','SentenceClass','PronounClass']).size())





#fig, ax = plt.subplots()
#for g in np.unique(gp_df["PronounNum"]):
#    i = np.where(gp_df["PronounNum"] == g)
#    print(j)
#    ax.scatter(gp_df["offsetPronoun-A"][j], gp_df["offsetPronoun-B"][j], c=g)
#ax.legend()
#ax.grid(True)
#fig.savefig('scatter.png')
#plt.close(fig)

