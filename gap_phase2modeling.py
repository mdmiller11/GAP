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

def divideSents(text, pronounOffset):
    offset = 0
    sentList = sent_tokenize(text)
    precedingText = ""
    sentWPronoun = ""
    succeedingText = ""
    foundSWP = False
    for i in sentList:
        if pronounOffset < len(str(i))+offset and foundSWP == False:
            sentWPronoun = i
            foundSWP == True
        elif foundSWP == False:
            precedingText += i + " "
        else:
            succeedingText += i + " "
        offset+=len(str(i))
    return precedingText, sentWPronoun, succeedingText
        
        
gp_df = pd.read_csv('gap_phase1-2.tsv', delimiter='\t')
gp_df["SentenceWithPronoun"] = gp_df["Pronoun"]
gp_df["PrecedingText"] = gp_df["Pronoun"]
gp_df["SucceedingText"] = gp_df["Pronoun"]

a=gp_df["Text"].iloc[1]
off=gp_df["Pronoun-offset"].iloc[1]

for i in range(len(gp_df['SentenceWithPronoun'])):
    a=gp_df["Text"].iloc[i]
    off=gp_df["Pronoun-offset"].iloc[i]
    gp_df["PrecedingText"].iloc[i], gp_df["SentenceWithPronoun"].iloc[i], gp_df["SucceedingText"].iloc[i] = divideSents(a, off)


print(gp_df.iloc[1])
gp_df.to_csv('gap_phase2_wDivSents.tsv', sep='\t', encoding='utf-8')

