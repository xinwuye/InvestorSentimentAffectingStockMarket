import pandas as pd
import numpy as np
import os
import pickle
from dict_sa import *
from segmentation import *
from data_handler import *
from sa_total import *
from Config import *
from mtrcs import *
from ml import *

text = pd.read_csv(os.path.join(path_of_results, 'text_finance_dict_sent_rid_nonscore.csv'), index_col=0)  # 已经prerocess过
# 与手工标注的1600条数据作差集
text_handmade = pd.read_excel(os.path.join(path, 'sample1600in16w.xlsx'), index_col=0)
text_handmade = preprocess(text_handmade)
text_handmade['发帖时间'] = text.发帖时间.map(lambda x: str(x)[: 10])
text = text.append(text_handmade).drop_duplicates(
        subset=['标题链接', '阅读', '评论', '发帖时间', '作者', '详情标题', '帖子内容'], keep=False)
# 去空值
text.dropna(subset=['content_seg_nonstop', 'title_seg_nonstop'], inplace=True)
# 重新设置index
text.reset_index(drop=True, inplace=True)
# 把空格分开的分词文本转换为列表
text['title_seg_nonstop_list'] = text.title_seg_nonstop.apply(lambda x: x.split())
with open(os.path.join(path_of_ml, 'ml_model_tfidf', 'title_seg_nonstop.pickle'), 'rb') as f:
    vectorizer = pickle.load(f)
X = text.title_seg_nonstop_list.to_list()
X = vectorizer.transform(X)

with open(os.path.join(path_of_ml, 'ml_model', 'title_seg_nonstop_LinearSVC.pickle'), 'rb') as f:
    clf = pickle.load(f)

preds = clf.predict(X)
text['sent'] = preds

# 2表示负面，转换成0
text_handmade['score_handmade'] = text_handmade.score_handmade.map(lambda x: x if x != 2.0 else 0)
text_handmade.columns
text_handmade.rename(columns={'score_handmade': 'sent'}, inplace=True)
text_final = pd.concat([text, text_handmade], ignore_index=True, join='inner')

# 输出，情感分析大功告成
text_final.to_csv(os.path.join(path_of_ml, 'sa_final_result.csv'))

# 构建情绪指数
sent = pd.read_csv(os.path.join(path_of_ml, 'sa_final_result.csv'), index_col=0,
                   usecols=['Unnamed: 0', '阅读', '评论', '发帖时间', 'sent'],
                   dtype={'发帖时间': str, '阅读': str, '评论': str, 'sent': float})
sent.rename(columns={'发帖时间': 'date'}, inplace=True)
print(sent.date.unique())
sent['date'] = sent['date'].map(lambda x: x[5:])
# 情感转换成1和-1
sent['sent'] = sent.sent.map(lambda x: 1 if x == 1 else -1)
# 看下时间范围对不对，应该是0506-0809
date = sent.date.unique()
print(date)
everyday = sent.groupby('date').sent.mean()
moods = pd.DataFrame(everyday).rename(columns={'sent': 'avg'})
# 按照各种论文里的方法构造各种情绪指数
# 基于股评的投资者情绪对中国股市的影响研究
# 各个帖子权重均为1计算得到的C_t_em
moods['c_t_pos'] = sent.groupby('date').sent.apply(lambda x: sum(x == 1))
moods['c_t_neg'] = sent.groupby('date').sent.apply(lambda x: sum(x == -1))
# B_t
moods['B_t'] = (moods.c_t_pos - moods.c_t_neg) / (moods.c_t_pos + moods.c_t_neg)
# B_t*
moods['B_t_star'] = moods.B_t * np.log(1 + moods.c_t_pos + moods.c_t_neg)
# 股吧舆情对股票市场的影响研究
lamb = 0.2
# 将阅读量和评论量从str转化为float
sent['read'] = sent['阅读'].map(transform)
sent['comment'] = sent['评论'].map(transform)
# 权重
sent['w'] = lamb * sent['read'] + (1 - lamb) * sent['comment']
sent['weighted_emotion'] = sent.w * sent.sent
# MTsell MTbuy
moods['MTbuy'] = sent.groupby('date').apply(lambda x: sum(x[x.weighted_emotion > 0].weighted_emotion))
moods['MTsell'] = sent.groupby('date').apply(lambda x: -sum(x[x.weighted_emotion < 0].weighted_emotion))
# P_T
moods['PT'] = (moods.MTbuy - moods.MTsell) / (moods.MTbuy + moods.MTsell)
# 指数
perf399976 = pd.read_excel(os.path.join(path, '399976perf.xlsx'))
perf399976['date'] = perf399976.日期Date.map(lambda x: str(x)[4:6] + '-' + str(x)[-2:])
# perf399976 = perf399976[(perf399976.date <= moods.index.max()) & (perf399976.date >= moods.index.min())]

moods = moods.merge(perf399976, on='date', how='left')
# moods.drop(moods.columns[11: 17], axis=1, inplace=True)
moods.rename(columns={'开盘Open': 'Open',
                      '最高High': 'High',
                      '最低Low': 'Low',
                      '收盘Close': 'Close',
                      '涨跌Change': 'Change',
                      '涨跌幅(%)Change(%)': 'Change_percent',
                      '成交量（万手）Volume(100M Shares)': 'Volume',
                      '成交金额（亿元）Turnover': 'Turnover',
                      '样本数量ConsNumber': 'ConsNumber'},
             inplace=True)
print(moods)
# 填充缺失值
fill_vars = ['Open', 'High', 'Low', 'Close', 'Change', 'Change_percent', 'Volume', 'Turnover', 'ConsNumber']
index_series = pd.Series(moods.index)
for var in fill_vars:
    moods[var] = index_series.apply(fill_individually, args=(moods[var], ))

# 把没用的列丢掉
moods.drop(labels=['日期Date', '指数代码Index Code', 'avg', 'c_t_pos', 'c_t_neg', 'MTbuy', 'MTsell', 'ConsNumber',
                   '指数中文全称Index Chinese Name(Full)',
                   '指数中文简称Index Chinese Name', '指数英文全称Index English Name(Full)',
                   '指数英文简称Index Chinese Name'],
           axis=1, inplace=True)
print('process done')
moods.to_csv(os.path.join(path_of_results, 'sent_stock_idxes.csv'))



