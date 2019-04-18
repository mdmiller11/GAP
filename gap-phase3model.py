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
from sklearn import tree
from scipy.sparse import coo_matrix, hstack, vstack
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier


gp_df = pd.read_csv('gap_phase2_wCV.tsv', delimiter='\t')

numerical_features = gp_df.loc[:,['Pronoun-offset', 'A-offset', 'B-offset', 
                                  'offsetPronoun-A', 'offsetPronoun-B', 'pre_predict-A', 
                                  'pre_predict-B', 'pre_predict-N', 'tar_predict-A', 
                                  'tar_predict-B', 'tar_predict-N', ]]
cat_features = pd.get_dummies(gp_df.loc[:,['PronounNum', 'SentenceClass']].astype('category'),drop_first=True)
labels = gp_df.loc[:, 'PronounClass']
ids = gp_df.loc[:, 'ID']
features = pd.concat([numerical_features,cat_features], axis=1)

clfANN = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
           beta_1=0.95, beta_2=0.9995, early_stopping=False,
           epsilon=1e-05, hidden_layer_sizes=(100,100),
           learning_rate='constant', learning_rate_init=0.015,
           max_iter=3000, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=0,
           shuffle=True, solver='adam', tol=0.001,
           validation_fraction=0.1, verbose=False, warm_start=False)

clfANN.fit(features, labels)
results = pd.DataFrame(clfANN.predict_proba(features))

finaldf = pd.concat([ids,results], axis=1)

finaldf = finaldf.rename(index=str, columns={0: "A", 1:"B", 2:"NEITHER"})
print(finaldf.iloc[1])

finaldf.to_csv('submission1.csv', index=False, sep=',', encoding='utf-8')
