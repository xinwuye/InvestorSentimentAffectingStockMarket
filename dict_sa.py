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

# def sa(text):
#     """
#     用姚加权,冯绪,王赞钧,纪荣嵘,张维.语调、情绪及市场影响:基于金融情绪词典[J].管理科学学报,2021,24(05):26-46.金融词典对单个文本进行情感分析
#     每个sentence的得分是正面得分-负面得分的标准化后结果，范围在-1到1之间
#     得分为正值的为正面文本，负反之
#     分值绝对值越接近1说明分类正确的概率越高
#     :param text: type str,用于情感分析的一段文本
#     :return: {'score': prob, 'word_num': word_num, 'stopword_num': stopword_num, 'pos_score': pos_score, 'neg_score': neg_score}
#     """
#     result = cntext.senti_by_finance(text, adj_adv=True)
#     prob = (result['pos_score'] - result['neg_score']) / (result['pos_score'] + result['neg_score'])
#     # 正负情感词总个数
#     # 这个包word_num的计算方法是wordnum = len(jieba.lcut(text))，所以这个word_num里面是包括stopword
#     # 所以下面输出的正负情感词个数应该为减去stopword_num
#     word_num = result['word_num']
#     stopword_num = result['stopword_num']
#     pos_score = result['pos_score']
#     neg_score = result['neg_score']
#     return {'score': prob, 'word_num': word_num, 'stopword_num': stopword_num, 'pos_score': pos_score, 'neg_score': neg_score}


def df_sa(text, pos=None, neg=None):
    '''
    对含有标题和内容的df进行情感分析并返回df类型的结果
    :param neg:
    :param pos:
    :param text:
    :return: 经过标题和帖子内容情感分析的text dataframe
    '''
    # 帖子内容情感分析
    content_result = text.帖子内容.apply(sa_diy, args=(pos, neg))
    print('帖子内容情感分析完成')
    text['content_score'] = content_result.map(lambda x: x['score'])
    text['content_word_num'] = content_result.map(lambda x: x['word_num'])
    text['content_stopword_num'] = content_result.map(lambda x: x['stopword_num'])
    text['content_pos_score'] = content_result.map(lambda x: x['pos_score'])
    text['content_neg_score'] = content_result.map(lambda x: x['neg_score'])
    # 详情标题情感分析
    title_result = text.详情标题.map(sa_diy)
    print('详情标题情感分析完成')
    text['title_score'] = title_result.map(lambda x: x['score'])
    text['title_word_num'] = title_result.map(lambda x: x['word_num'])
    text['title_stopword_num'] = title_result.map(lambda x: x['stopword_num'])
    text['title_pos_score'] = title_result.map(lambda x: x['pos_score'])
    text['title_neg_score'] = title_result.map(lambda x: x['neg_score'])
    # 输出一下正负分别的数量
    print('帖子内容负面数量：', sum(text.content_score < 0))
    print('帖子内容正面数量：', sum(text.content_score > 0))
    print('详情标题负面数量：', sum(text.title_score < 0))
    print('详情标题正面数量：', sum(text.title_score > 0))
    # 得分比例
    for thr in np.linspace(0.5, 0.95, round((0.95 - 0.5) / 0.05) + 1):
        tot_num = len(text['content_score'][text['content_score'].abs() > thr])
        pos_num = len(text['content_score'][text['content_score'] > thr])
        pos_neg = pos_num / (tot_num - pos_num)
        print('帖子内容情感得分绝对值大于%f的文本数:' % thr, tot_num)
        print('帖子内容情感得分绝对值大于%f的文本比例:（正：负）' % thr, pos_neg,
              '个数分别为：正：', pos_num, '负：', tot_num - pos_num)

    for thr in np.linspace(0.5, 0.95, round((0.95 - 0.5) / 0.05) + 1):
        tot_num = len(text['title_score'][text['title_score'].abs() > thr])
        pos_num = len(text['title_score'][text['title_score'] > thr])
        pos_neg = pos_num / (tot_num - pos_num)
        print('详情标题情感得分绝对值大于%f的文本数:' % thr, tot_num)
        print('详情标题情感得分绝对值大于%f的文本比例:（正：负）' % thr, pos_neg,
              '个数分别为：正：', pos_num, '负：', tot_num - pos_num)

    for thr in np.linspace(0.5, 0.95, round((0.95 - 0.5) / 0.05) + 1):
        pos_num = len(text['content_score'][(text['content_score'] > thr) & (text['title_score'] > thr)])
        neg_num = len(text['content_score'][(text['content_score'] < -thr) & (text['title_score'] < -thr)])
        tot_num = pos_num + neg_num
        pos_neg = pos_num / neg_num
        print('帖子内容和详情标题情感得分均大于%f或均小于-%f的文本数:' % (thr, thr), tot_num)
        print('帖子内容和详情标题情感得分均大于%f或均小于-%f的文本比例:（正：负）' % (thr, thr), pos_neg,
              '个数分别为：正：', pos_num, '负：', neg_num)

    return text


def sa_diy(text, pos=None, neg=None):
    '''
    自定义情感词典情感分析
    :param text: 一条文本字符串
    :param pos: 正面情感词典
    :param neg: 负面情感词典
    :return:
    '''
    # 每个sentence的得分是正面得分-负面得分的标准化后结果，范围在-1到1之间
    # 得分为正值的为正面文本，负反之
    # 分值绝对值越接近1说明分类正确的概率越高
    result = cntext.senti_by_finance(text, adj_adv=True, pos=pos, neg=neg)
    prob = (result['pos_score'] - result['neg_score']) / (result['pos_score'] + result['neg_score'])
    # 正负情感词总个数
    # 这个包word_num的计算方法是wordnum = len(jieba.lcut(text))，所以这个word_num里面是包括stopword
    # 所以下面输出的正负情感词个数应该为减去stopword_num
    word_num = result['word_num']
    stopword_num = result['stopword_num']
    pos_score = result['pos_score']
    neg_score = result['neg_score']
    return {'score': prob, 'word_num': word_num, 'stopword_num': stopword_num, 'pos_score': pos_score,
            'neg_score': neg_score}
