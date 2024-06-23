import pandas as pd
from sklearn import metrics
from data_handler import *
from dict_sa import *
from Config import *


def mtrcs(y, preds):
    '''
    输入真实值和预测值，输出指标
    :param y: 真实值
    :param preds: 预测值
    :return: 包含评价指标的df
    '''
    acc = metrics.accuracy_score(y, preds)
    pos_precision = metrics.precision_score(y, preds, pos_label=1)
    pos_recall = metrics.recall_score(y, preds, pos_label=1)
    pos_f1_score = metrics.f1_score(y, preds, pos_label=1)
    neg_precision = metrics.precision_score(y, preds, pos_label=0)
    neg_recall = metrics.recall_score(y, preds, pos_label=0)
    neg_f1_score = metrics.f1_score(y, preds, pos_label=0)
    result = {'acc': [acc], 'pos_precision': [pos_precision], 'pos_recall': [pos_recall], 'pos_f1_score': [pos_f1_score],
              'neg_precision': [neg_precision], 'neg_recall': [neg_recall], 'neg_f1_score': [neg_f1_score]}
    result = pd.DataFrame(result)
    return result


def evaluate(df, pos=None, neg=None, prefix=None):
    # df = preprocess(df)
    df = df_sa(df, pos, neg)
    df['sent'] = df.title_score.map(lambda x: 1 if x > 0 else 0)
    df['content_sent'] = df.content_score.map(lambda x: 1 if x > 0 else 0)
    result = mtrcs(df.score_handmade, df.sent)
    # result = pd.concat([result, mtrcs(df.score_handmade, df.content_sent)])
    result = result.append(mtrcs(df.score_handmade, df.content_sent))
    # 帖子内容和详情标题的分类结果相同的比例，相同且正确的比例
    # same_ratio = []

    if prefix is None:
        result.set_index(pd.Index(['title', 'content']), inplace=True)
    else:
        result.set_index(pd.Index(['title' + prefix, 'content' + prefix]), inplace=True)
    return result

