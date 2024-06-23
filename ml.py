import pandas as pd
import numpy as np
import os
import cntext
import seaborn as sns
import jieba
import glob
import re
import os
from time import time
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.utils.extmath import density
from sklearn import svm
from sklearn import naive_bayes
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.utils import shuffle
from Config import *


def train_on_whole(clf, X, y, clf_name):
    clf.fit(X, y)
    # save model
    with open(os.path.join(path_of_ml, 'ml_model', clf_name + '.pickle'), 'wb') as f:
        pickle.dump(clf, f)

    preds = clf.predict(X)

    acc = metrics.accuracy_score(y, preds)
    pos_precision = metrics.precision_score(y, preds, pos_label=1)
    pos_recall = metrics.recall_score(y, preds, pos_label=1)
    pos_f1_score = metrics.f1_score(y, preds, pos_label=1)
    neg_precision = metrics.precision_score(y, preds, pos_label=0)
    neg_recall = metrics.recall_score(y, preds, pos_label=0)
    neg_f1_score = metrics.f1_score(y, preds, pos_label=0)

    return acc, pos_precision, pos_recall, pos_f1_score, neg_precision, neg_recall, neg_f1_score


def benchmark_clfs_on_train(X, y, text_type):
    vectorizer = TfidfVectorizer(analyzer='word',
                                 tokenizer=dummy_fun,
                                 preprocessor=dummy_fun,
                                 token_pattern=None)

    vectorizer.fit(X)
    # save tfidf
    with open(os.path.join(path_of_ml, 'ml_model_tfidf', text_type + '.pickle'), 'wb') as f:
        pickle.dump(vectorizer, f)
    X = vectorizer.transform(X)

    classifiers = [
        ('LinearSVC', svm.LinearSVC()),
        ('LogisticReg', LogisticRegression()),
        ('SGD', SGDClassifier()),
        ('MultinomialNB', naive_bayes.MultinomialNB()),
        ('KNN', KNeighborsClassifier()),
        ('DecisionTree', DecisionTreeClassifier()),
        ('RandomForest', RandomForestClassifier()),
        ('AdaBoost', AdaBoostClassifier(base_estimator=LogisticRegression()))
    ]

    cols = ['metrics', 'accuracy',  'pos_precision', 'pos_recall', 'pos_f1_score', 'neg_precision', 'neg_recall', 'neg_f1_score']
    scores = []
    for name, clf in classifiers:
        score = train_on_whole(clf, X, y, text_type + '_' + name)
        row = [name]
        row.extend(score)
        scores.append(row)

    df = pd.DataFrame(scores, columns=cols).T
    df.columns = df.iloc[0]
    df.drop(df.index[[0]], inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')

    return df


def KFold_validation(clf, X, y, clf_name):
    acc = []
    pos_precision, pos_recall, pos_f1_score = [], [], []
    neg_precision, neg_recall, neg_f1_score = [], [], []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for j, (train, test) in enumerate(kf.split(X)):
        X_train = [X[i] for i in train]
        X_test = [X[i] for i in test]
        y_train = [y[i] for i in train]
        y_test = [y[i] for i in test]

        # vectorizer = TfidfVectorizer(analyzer='word', tokenizer=lambda x : (w for w in x.split(' ') if w.strip()))

        vectorizer = TfidfVectorizer(analyzer='word',
                                     tokenizer=dummy_fun,
                                     preprocessor=dummy_fun,
                                     token_pattern=None)

        vectorizer.fit(X_train)
        X_train = vectorizer.transform(X_train)
        X_test = vectorizer.transform(X_test)

        clf.fit(X_train, y_train)
        # # save model
        # with open(os.path.join(path_of_ml, 'kf_model', clf_name + str(j) + '.pickle'), 'wb') as f:
        #     pickle.dump(clf, f)

        preds = clf.predict(X_test)

        acc.append(metrics.accuracy_score(y_test, preds))
        pos_precision.append(metrics.precision_score(y_test, preds, pos_label=1))
        pos_recall.append(metrics.recall_score(y_test, preds, pos_label=1))
        pos_f1_score.append(metrics.f1_score(y_test, preds, pos_label=1))
        neg_precision.append(metrics.precision_score(y_test, preds, pos_label=0))
        neg_recall.append(metrics.recall_score(y_test, preds, pos_label=0))
        neg_f1_score.append(metrics.f1_score(y_test, preds, pos_label=0))

    return (np.mean(acc), np.mean(pos_precision), np.mean(pos_recall), np.mean(pos_f1_score),
            np.mean(neg_precision), np.mean(neg_recall), np.mean(neg_f1_score))


def benchmark_clfs(X, y, text_type):
    # X, y = load_dataset_tokenized()

    classifiers = [
        ('LinearSVC', svm.LinearSVC()),
        ('LogisticReg', LogisticRegression()),
        ('SGD', SGDClassifier()),
        ('MultinomialNB', naive_bayes.MultinomialNB()),
        ('KNN', KNeighborsClassifier()),
        ('DecisionTree', DecisionTreeClassifier()),
        ('RandomForest', RandomForestClassifier()),
        ('AdaBoost', AdaBoostClassifier(base_estimator=LogisticRegression()))
    ]

    cols = ['metrics', 'accuracy',  'pos_precision', 'pos_recall', 'pos_f1_score',
            'neg_precision', 'neg_recall', 'neg_f1_score']
    scores = []
    for name, clf in classifiers:
        score = KFold_validation(clf, X, y, text_type + '_' + name)
        row = [name]
        row.extend(score)
        scores.append(row)

    df = pd.DataFrame(scores, columns=cols).T
    df.columns = df.iloc[0]
    df.drop(df.index[[0]], inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')

    return df


def dummy_fun(doc):
    return doc


def test_on_handmade(clf, X, y):
    preds = clf.predict(X)

    acc = metrics.accuracy_score(y, preds)
    pos_precision = metrics.precision_score(y, preds, pos_label=1)
    pos_recall = metrics.recall_score(y, preds, pos_label=1)
    pos_f1_score = metrics.f1_score(y, preds, pos_label=1)
    neg_precision = metrics.precision_score(y, preds, pos_label=0)
    neg_recall = metrics.recall_score(y, preds, pos_label=0)
    neg_f1_score = metrics.f1_score(y, preds, pos_label=0)

    return acc, pos_precision, pos_recall, pos_f1_score, neg_precision, neg_recall, neg_f1_score


def benchmark_clfs_test(X, y, text_type):
    fileNames = glob.glob(path_of_ml + r'//ml_model//*')

    # 选定读入的模型
    if text_type == 'title_seg_nonstop':
        fileNames = np.array(fileNames)[list(map(lambda x: 'title_seg_nonstop' in x, fileNames))]
    elif text_type == 'content_seg_nonstop':
        fileNames = np.array(fileNames)[list(map(lambda x: 'content_seg_nonstop' in x, fileNames))]
    elif text_type == 'title_seg':
        fileNames = np.array(fileNames)[list(map(lambda x: ('title_seg' in x) and ('nonstop' not in x), fileNames))]
    elif text_type == 'content_seg':
        fileNames = np.array(fileNames)[list(map(lambda x: ('content_seg' in x) and ('nonstop' not in x), fileNames))]

    cols = ['metrics', 'accuracy',  'pos_precision', 'pos_recall',
            'pos_f1_score', 'neg_precision', 'neg_recall', 'neg_f1_score']
    scores = []

    with open(path_of_ml + '//ml_model_tfidf//' + text_type + '.pickle', 'rb') as f:
        vectorizer = pickle.load(f)
    X = vectorizer.transform(X)

    for fileName in fileNames:
        name = re.split('\\\\', fileName)[1].split('.')[0]
        with open(fileName, 'rb') as f:
            clf = pickle.load(f)
        score = test_on_handmade(clf, X, y)
        row = [name]
        row.extend(score)
        scores.append(row)

    df = pd.DataFrame(scores, columns=cols).T
    df.columns = df.iloc[0]
    df.drop(df.index[[0]], inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')

    return df
