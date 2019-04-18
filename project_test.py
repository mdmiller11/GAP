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




gp_df = pd.read_csv('gap-development.tsv', delimiter='\t')
gp_df["Pronoun_binary"] = gp_df["Pronoun"]

#gp_df.loc[gp_df.Pronoun == 'He','Pronoun_binary'] = 1

for i in range(len(gp_df['Pronoun'])):
    if gp_df['Pronoun'].iloc[i] == 'He' or gp_df['Pronoun'].iloc[i] == 'he':
        gp_df["Pronoun_binary"].iloc[i] = 0
    elif gp_df['Pronoun'].iloc[i] == 'She' or gp_df['Pronoun'].iloc[i] == 'she':
        gp_df["Pronoun_binary"].iloc[i] = 1
    elif gp_df['Pronoun'].iloc[i] == 'His' or gp_df['Pronoun'].iloc[i] == 'his':
        gp_df["Pronoun_binary"].iloc[i] = 2
    elif gp_df['Pronoun'].iloc[i] == 'Him' or gp_df['Pronoun'].iloc[i] == 'him':
        gp_df["Pronoun_binary"].iloc[i] = 3        
    elif gp_df['Pronoun'].iloc[i] == 'Her' or gp_df['Pronoun'].iloc[i] == 'her':
        gp_df["Pronoun_binary"].iloc[i] = 4 
    else:
        gp_df["Pronoun_binary"].iloc[i] = 5

print(gp_df.groupby('Pronoun_binary')['Pronoun_binary'].count())

print(gp_df.head())

gp_df["num_words"] = gp_df["Text"].apply(lambda x: len(str(x).split()))
gp_df["num_unique_words"] = gp_df["Text"].apply(lambda x: len(set(str(x).split())))
gp_df["num_chars"] = gp_df["Text"].apply(lambda x: len((str(x))))
gp_df["sentence_list"] = gp_df["Text"].apply(lambda x: sent_tokenize(str(x)))
gp_df.to_csv('gap.tsv', sep='\t', encoding='utf-8')


