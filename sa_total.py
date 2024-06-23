import pandas as pd
import numpy as np
import os
from dict_sa import *
from segmentation import *
from data_handler import *
from Config import *


def sa_total(text, pos=None, neg=None, prefix=None):
    # 情感分析
    text = df_sa(text, pos=pos, neg=neg)
    # 存一下
    if prefix is None:
        text.to_csv(os.path.join(path_of_results, 'text_finance_dict_sent_total5189.csv'))
    else:
        text.to_csv(os.path.join(path_of_results, prefix + '_text_finance_dict_sent_total5189.csv'))
    print('对所有样本情感分析完成')
    # 去掉不包含情感词典中词汇的样本
    text.dropna(subset=['content_score', 'title_score'], inplace=True)
    # 分词并去除停用词
    # 停用词表
    stopwords = cntext.dictionary.STOPWORDS_zh
    text['content_seg_nonstop'] = text.帖子内容.apply(seg_depart, args=[stopwords])
    text['title_seg_nonstop'] = text.详情标题.apply(seg_depart, args=[stopwords])
    print('分词完成')
    # 重新设置index
    text.reset_index(drop=True, inplace=True)
    if prefix is None:
        text.to_csv(os.path.join(path_of_results, 'text_finance_dict_sent_rid_nonscore.csv'))
    else:
        text.to_csv(os.path.join(path_of_results, prefix + 'text_finance_dict_sent_rid_nonscore.csv'))
    print('去除不包含情感词的样本并保存完成')
    # 保存去除了测试集的数据
    samp = pd.read_excel(os.path.join(path, 'sample1600in16w.xlsx'), index_col=0,
                         dtype={'content_seg': str, 'content_seg_nonstop': str,
                                'title_seg': str, 'title_seg_nonstop': str})
    print('训练集的大小为' + str(len(samp)))
    print('未去除测试集的训练集大小为' + str(len(text)))
    text = text.append(samp).drop_duplicates(
        subset=['标题链接', '阅读', '评论', '发帖时间', '作者', '详情标题', '帖子内容'], keep=False)
    print('去除测试集的训练集大小为' + str(len(text)))
    if prefix is None:
        text.to_csv(os.path.join(path_of_results, 'all_useful.csv'))
    else:
        text.to_csv(os.path.join(path_of_results, prefix + 'all_useful.csv'))
    # 筛选
    lam = 0.95
    text_filtered095 = text[((text['content_score'] > lam) & (text['title_score'] > lam)) |
                            ((text['content_score'] < -lam) & (text['title_score'] < -lam))]
    if prefix is None:
        text_filtered095.to_csv(os.path.join(path_of_results, 'train.csv'))
    else:
        text_filtered095.to_csv(os.path.join(path_of_results, prefix + 'train.csv'))


