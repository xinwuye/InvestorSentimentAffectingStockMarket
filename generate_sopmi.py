from harvesttext import HarvestText
import pandas as pd
import numpy as np
import os
import jieba
import cntext
import pickle
from Config import *
from segmentation import *
from data_handler import *
ht = HarvestText()

# 读入数据
text = pd.read_csv(os.path.join(path, 'merged_text.csv'), index_col=0)
# 截取5.1-8.9期间
text.dropna(subset=['发帖时间'], inplace=True)
text['发帖时间'] = original_to_datetime(text.发帖时间)
text = text[(text.发帖时间 >= pd.to_datetime('20210501', format='%Y%m%d')) &
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
# 分词并去除停用词
# 停用词表
stopwords = cntext.dictionary.STOPWORDS_zh
text['content_seg_nonstop'] = text.帖子内容.apply(seg_depart, args=[stopwords])
text['title_seg_nonstop'] = text.详情标题.apply(seg_depart, args=[stopwords])
print('分词完成')
text.dropna(subset=['content_seg_nonstop', 'title_seg_nonstop'], inplace=True)
# 把空格分开的分词文本转换为列表
text['content_seg_nonstop_list'] = text.content_seg_nonstop.apply(lambda x: x.split())
text['title_seg_nonstop_list'] = text.title_seg_nonstop.apply(lambda x: x.split())

pos_seeds = list(set(cntext.dictionary.FORMAL_pos_words) | set(cntext.dictionary.UNFORMAL_pos_words))
neg_seeds = list(set(cntext.dictionary.FORMAL_neg_words) | set(cntext.dictionary.UNFORMAL_neg_words))

for min_times in min_times_list:
    print('开始min_times=' + str(min_times))
    sent_dict_finance = ht.build_sent_dict(text.content_seg_nonstop_list.append(text.title_seg_nonstop_list),
                                           min_times=min_times,
                                           pos_seeds=pos_seeds,
                                           neg_seeds=neg_seeds,
                                           scale="+-1",
                                           sents_is_list=True)
    df_sent_dict_finance = pd.Series(sent_dict_finance.items())
    # so-pmi扩展出来的
    pos_sopmi = df_sent_dict_finance[df_sent_dict_finance.map(lambda x: x[1] > 0)].map(lambda x: x[0])
    neg_sopmi = df_sent_dict_finance[df_sent_dict_finance.map(lambda x: x[1] < 0)].map(lambda x: x[0])
    # 跟原来的合并起来
    pos_merged = set(pos_seeds) | set(pos_sopmi)
    neg_merged = set(neg_seeds) | set(neg_sopmi)
    # 保存SOPMI生成出来的词典
    with open(os.path.join(path_of_sopmi, 'finance_sopmi_only_pos' + str(min_times) + '.pickle'), 'wb') as f:
        pickle.dump(pos_sopmi, f)
    with open(os.path.join(path_of_sopmi, 'finance_sopmi_only_neg' + str(min_times) + '.pickle'), 'wb') as f:
        pickle.dump(neg_sopmi, f)
    pd.Series(pos_sopmi).to_excel(os.path.join(path_of_sopmi, 'finance_sopmi_only_pos' + str(min_times) + '.xlsx'))
    pd.Series(neg_sopmi).to_excel(os.path.join(path_of_sopmi, 'finance_sopmi_only_neg' + str(min_times) + '.xlsx'))
    # 保存合并后的词典
    with open(os.path.join(path_of_dict, 'finance_sopmi_pos_merged' + str(min_times) + '.pickle'), 'wb') as f:
        pickle.dump(pos_merged, f)
    with open(os.path.join(path_of_dict, 'finance_sopmi_neg_merged' + str(min_times) + '.pickle'), 'wb') as f:
        pickle.dump(neg_merged, f)
    print('结束min_times=' + str(min_times))

