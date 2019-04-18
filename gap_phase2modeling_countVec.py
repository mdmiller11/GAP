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

gp_df = pd.read_csv('gap_phase2_wLemmatization.tsv', delimiter='\t')

stop_words=set(stopwords.words('english'))
stop_words=stop_words.union('nan')
#print(gp_df.iloc[0])
cv = CountVectorizer(stop_words=stop_words)
print(type(gp_df["lemmTextAll"].iloc[0]))
cv.fit(gp_df["lemmTextAll"])
gp_df["CVlemmTextPreceding"]=gp_df["lemmTextPreceding"].apply(lambda x: cv.transform([str(x)]))
gp_df["CVlemmTextTarget"]=gp_df["lemmTextTarget"].apply(lambda x: cv.transform([str(x)]))
#print(gp_df.iloc[1]["CVlemmTextTarget"])
gp_df["CVlemmTextSucceding"]=gp_df["lemmTextSucceeding"].apply(lambda x: cv.transform([str(x)]))

labels = np.asarray(gp_df["PronounClass"])

tc_target = vstack([gp_df["CVlemmTextTarget"][0],gp_df["CVlemmTextTarget"][1]])
tc_pre = vstack([gp_df["CVlemmTextPreceding"][0],gp_df["CVlemmTextPreceding"][1]])
tc_suc = vstack([gp_df["CVlemmTextSucceding"][0],gp_df["CVlemmTextSucceding"][1]])
for i in range(2, len(gp_df["CVlemmTextTarget"])):
    tc_target = vstack([tc_target, gp_df["CVlemmTextTarget"][i]])
    tc_pre = vstack([tc_pre, gp_df["CVlemmTextPreceding"][i]])
    tc_suc = vstack([tc_suc, gp_df["CVlemmTextSucceding"][i]])
    
clf_SVM_Tar = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf_SVM_Pre = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

clf_SVM_Tar.fit(tc_target, labels)
clf_SVM_Pre.fit(tc_pre, labels)


gp_df["pre_predict-A"] = gp_df["CVlemmTextPreceding"].apply(lambda x: clf_SVM_Pre.predict_proba(x)[0][0])
gp_df["pre_predict-B"] = gp_df["CVlemmTextPreceding"].apply(lambda x: clf_SVM_Pre.predict_proba(x)[0][1])
gp_df["pre_predict-N"] = gp_df["CVlemmTextPreceding"].apply(lambda x: clf_SVM_Pre.predict_proba(x)[0][2])
gp_df["tar_predict-A"] = gp_df["CVlemmTextPreceding"].apply(lambda x: clf_SVM_Tar.predict_proba(x)[0][0])
gp_df["tar_predict-B"] = gp_df["CVlemmTextPreceding"].apply(lambda x: clf_SVM_Tar.predict_proba(x)[0][1])
gp_df["tar_predict-N"] = gp_df["CVlemmTextPreceding"].apply(lambda x: clf_SVM_Tar.predict_proba(x)[0][2])

gp_df.to_csv('gap_phase2_wCV.tsv', sep='\t', encoding='utf-8')
