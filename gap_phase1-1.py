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

gp_df = pd.read_csv('gap.tsv', delimiter='\t')
#gp_df["wordsFiltered"] = gp_df["Text"].apply(lambda x: words_filter_stopwords(word_tokenize(str(x))))
gp_df["offsetPronoun-A"]=gp_df["A-offset"]-gp_df["Pronoun-offset"]
gp_df["offsetPronoun-B"]=gp_df["B-offset"]-gp_df["Pronoun-offset"]
gp_df["PronounClass"]=gp_df["Pronoun"]
for i in range(len(gp_df['Pronoun'])):
    if gp_df['A-coref'].iloc[i] == True:
        gp_df["PronounClass"].iloc[i] = "A"
    elif gp_df['B-coref'].iloc[i] == True:
        gp_df["PronounClass"].iloc[i] = "B"
    else:
        gp_df["PronounClass"].iloc[i] = "Neither"
        
gp_df["SentenceClass"]=gp_df["Pronoun"]
for i in range(len(gp_df['Pronoun'])):
    if gp_df['Pronoun-offset'].iloc[i] <= gp_df['A-offset'].iloc[i]:
        gp_df["SentenceClass"].iloc[i] = "InitialPro"
    elif gp_df['Pronoun-offset'].iloc[i] <= gp_df['B-offset'].iloc[i]:
        gp_df["SentenceClass"].iloc[i] = "MiddlePro"
    else:
        gp_df["SentenceClass"].iloc[i] = "FinalPro"

print(gp_df.groupby('PronounClass')['PronounClass'].count())
print(gp_df.groupby('Pronoun_binary')['Pronoun_binary'].count())

print(gp_df.groupby(['SentenceClass','PronounClass']).size())
gp_df["PronounNum"]=gp_df["PronounClass"]

for i in range(len(gp_df['PronounNum'])):
    if gp_df['PronounClass'].iloc[i] == "A":
        gp_df["PronounNum"].iloc[i] = 1
    elif gp_df['PronounClass'].iloc[i] == "B":
        gp_df["PronounNum"].iloc[i] = 2
    else:
        gp_df["PronounNum"].iloc[i] = 3

gp_df.to_csv('gap_phase1-2.tsv', sep='\t', encoding='utf-8')

#fig, ax = plt.subplots()
#for g in np.unique(gp_df["PronounNum"]):
#    i = np.where(gp_df["PronounNum"] == g)
#    j = pd.MultiIndex.from_tuples(i)
#    ax.scatter(gp_df["offsetPronoun-A"][j], gp_df["offsetPronoun-B"][j], c=g)
#ax.legend()
#ax.grid(True)
#fig.savefig('scatter.png')
#plt.close(fig)
#print(gp_df.loc[2])








