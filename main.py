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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 读入数据
    text_for_eval = pd.read_excel(os.path.join(path, 'sample1600in16w.xlsx'), index_col=0,
                                  dtype={'content_seg': str, 'content_seg_nonstop': str,
                                         'title_seg': str, 'title_seg_nonstop': str})
    text_for_eval = preprocess(text_for_eval)
    # 2表示负面，转换成0
    text_for_eval['score_handmade'] = text_for_eval.score_handmade.map(lambda x: x if x != 2.0 else 0)
    # 去掉还没标注的, 以及分词后为空的
    text_for_eval.dropna(subset=['score_handmade', 'content_seg', 'content_seg_nonstop',
                                 'title_seg', 'title_seg_nonstop'], inplace=True)

    # 基于词典的情感分析的评价指标
    print('开始输出基于词典的情感分析的评价指标')
    # 用原始情感词典做情感分析
    eval_result = evaluate(text_for_eval.copy())
    print(eval_result)
    title_result = eval_result.iloc[0].to_frame().T
    content_result = eval_result.iloc[1].to_frame().T
    # # 用SOPMI出来的情感词典做情感分析
    # for min_times in min_times_list:
    #     with open(os.path.join(path_of_dict, 'finance_sopmi_pos_merged' + str(min_times) + '.pickle'), 'rb') as f:
    #         pos = pickle.load(f)
    #     with open(os.path.join(path_of_dict, 'finance_sopmi_neg_merged' + str(min_times) + '.pickle'), 'rb') as f:
    #         neg = pickle.load(f)
    #     # eval_result = eval_result.append(evaluate(text_for_eval.copy(), pos, neg, prefix=str(min_times)))
    #     eval_result = evaluate(text_for_eval.copy(), pos, neg, prefix=str(min_times))
    #     print(eval_result)
    #     title_result = title_result.append(eval_result.iloc[0].to_frame().T)
    #     content_result = content_result.append(eval_result.iloc[1].to_frame().T)
    title_result.to_excel(os.path.join(path_of_metrics, 'metrics_dict_sa_title.xlsx'))
    content_result.to_excel(os.path.join(path_of_metrics, 'metrics_dict_sa_content.xlsx'))
    print('结束输出基于词典的情感分析的评价指标')

    # 基于词典的情感分析，直接用原始的词典，无语了
    print('开始基于词典的情感分析')
    text = pd.read_csv(os.path.join(path, 'merged_text.csv'), index_col=0)
    text = preprocess(text)

    sa_total(text)
    print('基于词典的情感分析结束')
    # 机器学习
    print('开始ml')
    train = pd.read_csv(os.path.join(path_of_results, 'train.csv'), index_col=0)
    # train = preprocess(train)
    print('开始预处理')
    # 去空值
    train.dropna(subset=['content_seg_nonstop', 'title_seg_nonstop'], inplace=True)
    # 创建y标签列
    train['sent'] = train.title_score.map(lambda x: 1 if x > 0 else 0)
    # 把空格分开的分词文本转换为列表
    train['title_seg_nonstop_list'] = train.title_seg_nonstop.apply(lambda x: x.split())
    # 用标题文本、去除停用词训练
    # 转换成所需要的格式
    X_title_seg_nonstop = train.title_seg_nonstop_list.to_list()
    y = train.sent.to_list()
    print('结束预处理')
    # 5fold并输出各项指标
    print('开始 5fold')
    result_5fold_title_seg_nonstop = benchmark_clfs(X_title_seg_nonstop, y, 'title_seg_nonstop')
    result_5fold_title_seg_nonstop.to_excel(os.path.join(path_of_metrics, 'metrics_5fold_ml_sa_title.xlsx'))
    print('5fold finished.')
    # 在整个有情感倾向的数据集上训练并保存模型
    print('开始训练并保存模型')
    # 转换成所需要的格式
    X_title_seg_nonstop = train.title_seg_nonstop_list.to_list()
    y = train.sent.to_list()
    # 训练并输出各项指标
    result_title_seg_nonstop = benchmark_clfs_on_train(X_title_seg_nonstop, y, 'title_seg_nonstop')
    result_title_seg_nonstop.to_excel(os.path.join(path_of_metrics, 'metrics_train_ml_sa_title.xlsx'))
    print('train finished.')
    # 在测试集上测试
    print('开始在测试集上跑模型')
    # 把空格分开的分词文本转换为列表
    text_for_eval['title_seg_nonstop_list'] = text_for_eval.title_seg_nonstop.apply(lambda x: x.split())
    # 用标题文本、去除停用词训练
    # 转换成所需要的格式
    X_title_seg_nonstop = text_for_eval.title_seg_nonstop_list.to_list()
    y = text_for_eval.score_handmade.to_list()
    # 输出各项指标
    test_result_title_seg_nonstop = benchmark_clfs_test(X_title_seg_nonstop, y, 'title_seg_nonstop')
    test_result_title_seg_nonstop.to_excel(os.path.join(path_of_metrics, 'metrics_test_ml_sa_title.xlsx'))
    print('test finished.')

