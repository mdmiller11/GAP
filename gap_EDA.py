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


gp_df = pd.read_csv('gap_phase2.tsv', delimiter='\t')
print(gp_df.groupby(['Pronoun_binary','SentenceClass','PronounClass']).size())
print(gp_df.iloc[1])