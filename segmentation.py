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


def seg_depart(single_seg_text, stopwords):
    '''
    去除分词中的停用词
    对句子进行中文分词
    :param single_seg_text: 经过分词并用空格连接的文本字符串
    :param stopwords: 停用词字典
    :return:
    '''
    # 分词后用空格连接
    single_seg_text = ' '.join(jieba.cut(single_seg_text))
    # 去停用词
    text_list = single_seg_text.split()
    text_list = [w for w in text_list if w not in stopwords]
    # 用空格连接成字符串之后返回
    return ' '.join(text_list)

