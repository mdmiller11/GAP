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
from sklearn.model_selection import GridSearchCV


#@inproceedings{webster2018gap,
#  title =     {Mind the GAP: A Balanced Corpus of Gendered Ambiguous Pronouns},
#  author =    {Webster, Kellie and Recasens, Marta and Axelrod, Vera and Baldridge, Jason},
#  booktitle = {Transactions of the ACL},
#  year =      {2018},
#  pages =     {to appear},
#}

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

def lemmatize(text):
    _POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
    tagger = load(_POS_TAGGER)
    wnpos = lambda e: ('a' if e[0].lower() == 'j' else e[0].lower()) if e[0].lower() in ['n', 'r', 'v'] else 'n'
    lemmatizer = WordNetLemmatizer()

#    lemmatizedSummaryText=[]
    words = text.lower().strip().split(' ')
    tagged=tagger.tag(words)
    lem_sent_words = [lemmatizer.lemmatize(t[0],wnpos(t[1])) for t in tagged]
    lem_sentence=""
    for i in lem_sent_words:
        lem_sentence += i + ' '
#    lemmatizedSummaryText.append(lem_sentence)
#    return lemmatizedSummaryText
    return lem_sentence

def count_occurrences(word, sentence):
#    return re.sub('[^0-9A-Za-z]',' ',sentence).lower().split().count(word)
    return sentence.count(word)
train_df = pd.read_csv('gap-test.tsv', delimiter='\t')        ###Training Dataset from GAP

train_df["Pronoun_binary"] = train_df["Pronoun"]
train_df["PronounClass"]=train_df["Pronoun"]
train_df["SentenceClass"]=train_df["Pronoun"]
train_df["GenderBin"]=train_df["PronounClass"]
train_df["PronounType"]=train_df["PronounClass"]
train_df["SentenceWithPronoun"] = train_df["Pronoun"]
train_df["PrecedingText"] = train_df["Pronoun"]
train_df["SucceedingText"] = train_df["Pronoun"]
train_df["CountNounA"] = train_df["Pronoun"]
train_df["CountNounB"] = train_df["Pronoun"]
train_df["IsInitialPro"] = train_df["Pronoun"]
train_df["IsFinalPro"] = train_df["Pronoun"]

test_df = pd.read_csv('test_stage_2.tsv', delimiter='\t')       ###Test Dataset provided by Kaggle, Google AI
test_df["Pronoun_binary"] = test_df["Pronoun"]
#####train_df["PronounClass"]=train_df["Pronoun"] - Output var, do not add####
test_df["SentenceClass"]=test_df["Pronoun"]
test_df["GenderBin"]=test_df["Pronoun"]
test_df["PronounType"]=test_df["Pronoun"]
test_df["SentenceWithPronoun"] = test_df["Pronoun"]
test_df["PrecedingText"] = test_df["Pronoun"]
test_df["SucceedingText"] = test_df["Pronoun"]
test_df["CountNounA"] = test_df["Pronoun"]
test_df["CountNounB"] = test_df["Pronoun"]
test_df["IsInitialPro"] = test_df["Pronoun"]
test_df["IsFinalPro"] = test_df["Pronoun"]

pd.options.mode.chained_assignment = None  # default='warn'

for i in range(len(train_df['Pronoun'])):
    if train_df['Pronoun'].iloc[i] == 'He' or train_df['Pronoun'].iloc[i] == 'he':
        train_df["Pronoun_binary"].iloc[i] = 0
    elif train_df['Pronoun'].iloc[i] == 'She' or train_df['Pronoun'].iloc[i] == 'she':
        train_df["Pronoun_binary"].iloc[i] = 1
    elif train_df['Pronoun'].iloc[i] == 'His' or train_df['Pronoun'].iloc[i] == 'his':
        train_df["Pronoun_binary"].iloc[i] = 2
    elif train_df['Pronoun'].iloc[i] == 'Him' or train_df['Pronoun'].iloc[i] == 'him':
        train_df["Pronoun_binary"].iloc[i] = 3        
    elif train_df['Pronoun'].iloc[i] == 'Her' or train_df['Pronoun'].iloc[i] == 'her':
        train_df["Pronoun_binary"].iloc[i] = 4 
    else:
        train_df["Pronoun_binary"].iloc[i] = 5
        
    if train_df['A-coref'].iloc[i] == True:
        train_df["PronounClass"].iloc[i] = "A"
    elif train_df['B-coref'].iloc[i] == True:
        train_df["PronounClass"].iloc[i] = "B"
    else:
        train_df["PronounClass"].iloc[i] = "Neither"
        
    if train_df['Pronoun-offset'].iloc[i] <= train_df['A-offset'].iloc[i]:
        train_df["SentenceClass"].iloc[i] = "InitialPro"
    elif train_df['Pronoun-offset'].iloc[i] <= train_df['B-offset'].iloc[i]:
        train_df["SentenceClass"].iloc[i] = "MiddlePro"
    else:
        train_df["SentenceClass"].iloc[i] = "FinalPro"
    
    if train_df['SentenceClass'].iloc[i] == "InitialPro":    
        train_df["IsInitialPro"].iloc[i] = 1
        train_df["IsFinalPro"].iloc[i] = 0
    elif train_df['SentenceClass'].iloc[i] == "FinalPro":
        train_df["IsInitialPro"].iloc[i] = 0
        train_df["IsFinalPro"].iloc[i] = 1
    else:
        train_df["IsInitialPro"].iloc[i] = 0
        train_df["IsFinalPro"].iloc[i] = 0
        
    if train_df['Pronoun_binary'].iloc[i] == 0 or train_df['Pronoun_binary'].iloc[i] == 1:
        train_df["PronounType"].iloc[i] = 0 ##he/she
    else:
        train_df["PronounType"].iloc[i] = 1 ##his/her/him/her/hers
        
    if train_df['Pronoun_binary'].iloc[i] == 0 or train_df['Pronoun_binary'].iloc[i] == 2 or train_df['Pronoun_binary'].iloc[i] == 3:
        train_df["GenderBin"].iloc[i] = 0
    else:
        train_df["GenderBin"].iloc[i] = 1

    a=train_df["Text"].iloc[i]
    off=train_df["Pronoun-offset"].iloc[i]
    train_df["PrecedingText"].iloc[i], train_df["SentenceWithPronoun"].iloc[i], train_df["SucceedingText"].iloc[i] = divideSents(a, off)
    
    train_df["CountNounA"].iloc[i] = count_occurrences(train_df["A"].iloc[i], train_df["Text"].iloc[i])
    train_df["CountNounB"].iloc[i] = count_occurrences(train_df["B"].iloc[i], train_df["Text"].iloc[i])


train_df["offsetPronoun-A"]=train_df["A-offset"]-train_df["Pronoun-offset"]
train_df["offsetPronoun-B"]=train_df["B-offset"]-train_df["Pronoun-offset"]

stop_words=set(stopwords.words('english'))
stop_words=stop_words.union('nan')
train_cv = CountVectorizer(stop_words=stop_words)

train_df.to_csv('checkpoint.csv', index=False, sep=',', encoding='utf-8')

train_df["lemmTextAll"]=train_df["Text"].apply(lambda x: lemmatize(str(x)))
train_cv.fit(train_df["lemmTextAll"])

train_df["lemmTextPreceding"]=train_df["PrecedingText"].apply(lambda x: lemmatize(str(x)))
train_df["lemmTextTarget"]=train_df["SentenceWithPronoun"].apply(lambda x: lemmatize(str(x)))
#gp_df["lemmTextSucceeding"]=gp_df["SucceedingText"].apply(lambda x: lemmatize(str(x)))



train_df["CVlemmTextPreceding"]=train_df["lemmTextPreceding"].apply(lambda x: train_cv.transform([str(x)]))
train_df["CVlemmTextTarget"]=train_df["lemmTextTarget"].apply(lambda x: train_cv.transform([str(x)]))
#print(gp_df.iloc[1]["CVlemmTextTarget"])

train_labels = np.asarray(train_df["PronounClass"])

tc_target_train = vstack([train_df["CVlemmTextTarget"][0],train_df["CVlemmTextTarget"][1]])
tc_pre_train = vstack([train_df["CVlemmTextPreceding"][0],train_df["CVlemmTextPreceding"][1]])
for i in range(2, len(train_df["CVlemmTextTarget"])):
    tc_target_train = vstack([tc_target_train, train_df["CVlemmTextTarget"][i]])
    tc_pre_train = vstack([tc_pre_train, train_df["CVlemmTextPreceding"][i]])
    
clf_SVM_Tar = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf_SVM_Pre = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

svm_grid=[{'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
       'degree': [3,5]
        }]

gs_clf_pre=GridSearchCV(clf_SVM_Pre, svm_grid, cv=5)
gs_clf_tar=GridSearchCV(clf_SVM_Tar, svm_grid, cv=5)



gs_clf_pre.fit(tc_target_train, train_labels)
gs_clf_tar.fit(tc_pre_train, train_labels)


train_df["pre_predict-A"] = train_df["CVlemmTextPreceding"].apply(lambda x: gs_clf_pre.predict_proba(x)[0][0])
train_df["pre_predict-B"] = train_df["CVlemmTextPreceding"].apply(lambda x: gs_clf_pre.predict_proba(x)[0][1])
train_df["pre_predict-N"] = train_df["CVlemmTextPreceding"].apply(lambda x: gs_clf_pre.predict_proba(x)[0][2])
train_df["tar_predict-A"] = train_df["CVlemmTextPreceding"].apply(lambda x: gs_clf_tar.predict_proba(x)[0][0])
train_df["tar_predict-B"] = train_df["CVlemmTextPreceding"].apply(lambda x: gs_clf_tar.predict_proba(x)[0][1])
train_df["tar_predict-N"] = train_df["CVlemmTextPreceding"].apply(lambda x: gs_clf_tar.predict_proba(x)[0][2])

features_train = train_df.loc[:,['Pronoun-offset', 'A-offset', 'B-offset', 
                                  'offsetPronoun-A', 'offsetPronoun-B', 'pre_predict-A', 
                                  'pre_predict-B', 'pre_predict-N', 'tar_predict-A', 
                                  'tar_predict-B', 'tar_predict-N', 'PronounType', 'GenderBin', 'CountNounA', 'CountNounB', 'Pronoun_binary', 'IsFinalPro', 'IsInitialPro']]
#cat_features_train = pd.get_dummies(train_df.loc[:,['Pronoun_binary', 'SentenceClass']].astype('category'),drop_first=True)
ids_train = train_df.loc[:, 'ID']
#features_train = pd.concat([numerical_features_train,cat_features_train], axis=1)
features_train.to_csv('checkpoint_featuresTrain.csv', index=False, sep=',', encoding='utf-8')

clfANN = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
           beta_1=0.95, beta_2=0.9995, early_stopping=False,
           epsilon=1e-05, hidden_layer_sizes=(200,200),
           learning_rate='constant', learning_rate_init=0.01,
           max_iter=3000, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=0,
           shuffle=True, solver='adam', tol=0.001,
           validation_fraction=0.1, verbose=False, warm_start=False)

clfANN.fit(features_train, train_labels)

#############COMPLETED TRAINING###################

for i in range(len(test_df['Pronoun'])):
    if test_df['Pronoun'].iloc[i] == 'He' or test_df['Pronoun'].iloc[i] == 'he':
        test_df["Pronoun_binary"].iloc[i] = 0
    elif test_df['Pronoun'].iloc[i] == 'She' or test_df['Pronoun'].iloc[i] == 'she':
        test_df["Pronoun_binary"].iloc[i] = 1
    elif test_df['Pronoun'].iloc[i] == 'His' or test_df['Pronoun'].iloc[i] == 'his':
        test_df["Pronoun_binary"].iloc[i] = 2
    elif test_df['Pronoun'].iloc[i] == 'Him' or test_df['Pronoun'].iloc[i] == 'him':
        test_df["Pronoun_binary"].iloc[i] = 3        
    elif test_df['Pronoun'].iloc[i] == 'Her' or test_df['Pronoun'].iloc[i] == 'her':
        test_df["Pronoun_binary"].iloc[i] = 4 
    else:
        test_df["Pronoun_binary"].iloc[i] = 5
        
    if test_df['Pronoun-offset'].iloc[i] <= test_df['A-offset'].iloc[i]:
        test_df["SentenceClass"].iloc[i] = "InitialPro"
    elif test_df['Pronoun-offset'].iloc[i] <= test_df['B-offset'].iloc[i]:
        test_df["SentenceClass"].iloc[i] = "MiddlePro"
    else:
        test_df["SentenceClass"].iloc[i] = "FinalPro"
        
    if test_df['SentenceClass'].iloc[i] == "InitialPro":    
        test_df["IsInitialPro"].iloc[i] = 1
        test_df["IsFinalPro"].iloc[i] = 0
    elif test_df['SentenceClass'].iloc[i] == "FinalPro":
        test_df["IsInitialPro"].iloc[i] = 0
        test_df["IsFinalPro"].iloc[i] = 1
    else:
        test_df["IsInitialPro"].iloc[i] = 0
        test_df["IsFinalPro"].iloc[i] = 0

    if test_df['Pronoun_binary'].iloc[i] == 0 or test_df['Pronoun_binary'].iloc[i] == 1:
        test_df["PronounType"].iloc[i] = 0 ##he/she
    else:
        test_df["PronounType"].iloc[i] = 1 ##his/her/him/her/hers
        
    if test_df['Pronoun_binary'].iloc[i] == 0 or test_df['Pronoun_binary'].iloc[i] == 2 or test_df['Pronoun_binary'].iloc[i] == 3:
        test_df["GenderBin"].iloc[i] = 0
    else:
        test_df["GenderBin"].iloc[i] = 1

    a=test_df["Text"].iloc[i]
    off=test_df["Pronoun-offset"].iloc[i]
    test_df["PrecedingText"].iloc[i], test_df["SentenceWithPronoun"].iloc[i], test_df["SucceedingText"].iloc[i] = divideSents(a, off)
    
    test_df["CountNounA"].iloc[i] = count_occurrences(test_df["A"].iloc[i], test_df["Text"].iloc[i])
    test_df["CountNounB"].iloc[i] = count_occurrences(test_df["B"].iloc[i], test_df["Text"].iloc[i])

test_df["offsetPronoun-A"]=test_df["A-offset"]-test_df["Pronoun-offset"]
test_df["offsetPronoun-B"]=test_df["B-offset"]-test_df["Pronoun-offset"]

test_df["lemmTextPreceding"]=test_df["PrecedingText"].apply(lambda x: lemmatize(str(x)))
test_df["lemmTextTarget"]=test_df["SentenceWithPronoun"].apply(lambda x: lemmatize(str(x)))

test_df["CVlemmTextPreceding"]=test_df["lemmTextPreceding"].apply(lambda x: train_cv.transform([str(x)]))
test_df["CVlemmTextTarget"]=test_df["lemmTextTarget"].apply(lambda x: train_cv.transform([str(x)]))

test_df["pre_predict-A"] = test_df["CVlemmTextPreceding"].apply(lambda x: gs_clf_pre.predict_proba(x)[0][0])
test_df["pre_predict-B"] = test_df["CVlemmTextPreceding"].apply(lambda x: gs_clf_pre.predict_proba(x)[0][1])
test_df["pre_predict-N"] = test_df["CVlemmTextPreceding"].apply(lambda x: gs_clf_pre.predict_proba(x)[0][2])
test_df["tar_predict-A"] = test_df["CVlemmTextPreceding"].apply(lambda x: gs_clf_tar.predict_proba(x)[0][0])
test_df["tar_predict-B"] = test_df["CVlemmTextPreceding"].apply(lambda x: gs_clf_tar.predict_proba(x)[0][1])
test_df["tar_predict-N"] = test_df["CVlemmTextPreceding"].apply(lambda x: gs_clf_tar.predict_proba(x)[0][2])

features_test = test_df.loc[:,['Pronoun-offset', 'A-offset', 'B-offset', 
                                  'offsetPronoun-A', 'offsetPronoun-B', 'pre_predict-A', 
                                  'pre_predict-B', 'pre_predict-N', 'tar_predict-A', 
                                  'tar_predict-B', 'tar_predict-N', 'PronounType', 'GenderBin', 'CountNounA', 'CountNounB', 'Pronoun_binary', 'IsFinalPro', 'IsInitialPro']]
#cat_features_test = pd.get_dummies(test_df.loc[:,['Pronoun_binary', 'SentenceClass']].astype('category'),drop_first=True)
ids_test = test_df.loc[:, 'ID']
#features_test = pd.concat([numerical_features_test,cat_features_test], axis=1)
features_test.to_csv('checkpoint_featuresTest.csv', index=False, sep=',', encoding='utf-8')

results = pd.DataFrame(clfANN.predict_proba(features_test))

finaldf = pd.concat([ids_test,results], axis=1)

finaldf = finaldf.rename(index=str, columns={0: "A", 1:"B", 2:"NEITHER"})

finaldf.to_csv('submission.csv', index=False, sep=',', encoding='utf-8')

