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
from sklearn.feature_extraction.text import CountVectorizer


gp_df = pd.read_csv('gap_phase2_wDivSents.tsv', delimiter='\t')

stop_words=set(stopwords.words('english'))
representativeSentences = []
cv = CountVectorizer(stop_words=stop_words)

def lemmatize(text):
    _POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
    tagger = load(_POS_TAGGER)
    wnpos = lambda e: ('a' if e[0].lower() == 'j' else e[0].lower()) if e[0].lower() in ['n', 'r', 'v'] else 'n'
    lemmatizer = WordNetLemmatizer()

    lemmatizedSummaryText=[]
    words = text.lower().strip().split(' ')
    tagged=tagger.tag(words)
    lem_sent_words = [lemmatizer.lemmatize(t[0],wnpos(t[1])) for t in tagged]
    lem_sentence=""
    for i in lem_sent_words:
        lem_sentence += i + ' '
    lemmatizedSummaryText.append(lem_sentence)
    return lemmatizedSummaryText
print(gp_df.iloc[1])
gp_df["lemmTextAll"]=gp_df["Text"].apply(lambda x: lemmatize(str(x)))
gp_df["lemmTextPreceding"]=gp_df["PrecedingText"].apply(lambda x: lemmatize(str(x)))
gp_df["lemmTextTarget"]=gp_df["SentenceWithPronoun"].apply(lambda x: lemmatize(str(x)))
gp_df["lemmTextSucceeding"]=gp_df["SucceedingText"].apply(lambda x: lemmatize(str(x)))
gp_df.to_csv('gap_phase2_wLemmatization.tsv', sep='\t', encoding='utf-8')