import pandas as pd
import numpy as np
import re
import os
from Config import *


def opt(x):
    if ' ' in x:
        y = x[:3] + '0' + x[3]
    else:
        y = x
    return y


def transform(x):
    if '万' not in x:
        y = float(x)
    else:
        y = float(x[:-1]) * 10000
    return y


def original_to_datetime(date_series):
    date_series = date_series.map(lambda x: str(x)[:5]).map(opt)
    date_series = pd.to_datetime(date_series.map(lambda x: '2021-' + x))
    return date_series


# make Chinese text clean
def clean_zh_text(text):
    # Chinese
    comp = re.compile('[^\u4e00-\u9fa5]')
    return comp.sub('', text)


def preprocess(text):
    text.drop_duplicates(keep='first')
    # 截取5.6-8.9期间
    print('开始预处理')
    text.dropna(subset=['发帖时间'], inplace=True)
    text['发帖时间'] = original_to_datetime(text.发帖时间)
    text = text[(text.发帖时间 >= pd.to_datetime('20210506', format='%Y%m%d')) &
                (text.发帖时间 <= pd.to_datetime('20210809', format='%Y%m%d'))]
    # 去除帖子内容缺失的样本
    text.dropna(subset=['帖子内容', '详情标题'], inplace=True)
    # 去除非中文字符
    text['帖子内容'] = text.帖子内容.map(clean_zh_text)
    text['详情标题'] = text.详情标题.map(clean_zh_text)
    # 再次去除帖子内容缺失的样本
    text.dropna(subset=['帖子内容', '详情标题'], inplace=True)
    text.reset_index(drop=True, inplace=True)
    print('预处理完成')
    return text


# 用缺失值前一个和后一个非缺失值的平均值填充缺失值
def fill_individually(idx, srs):
    val = srs.iloc[idx, ]
    if np.isnan(val):
        # 找到前一个非零的值
        former_srs = srs.iloc[: idx].dropna()
        last = former_srs.iloc[-1]
        # 找到后一个非零的值
        later_srs = srs.iloc[idx: ].dropna()
        nxt = later_srs.iloc[0]
        # 判断这一缺失值所在的连续的缺失值串的长度
        last_idx = former_srs.index[-1]
        next_idx = later_srs.index[0]
        n = next_idx - last_idx - 1
        # 判断这一缺失值在这个连续的缺失值串中的位置
        location = idx - last_idx
        # 填充值
        ret = (nxt - last) / (n + 1) * location + last
    else:
        ret = val

    return ret


