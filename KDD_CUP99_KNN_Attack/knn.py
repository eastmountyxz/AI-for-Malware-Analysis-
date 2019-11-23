# coding=utf-8
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def classify(input_vct, data_set):
    data_set_size = data_set.shape[0]
    diff_mat = np.tile(input_vct, (data_set_size, 1)) - data_set  #扩充input_vct到与data_set同型并相减
    sq_diff_mat = diff_mat**2                                     #矩阵中每个元素都平方
    distance = sq_diff_mat.sum(axis=1)**0.5                       #每行相加求和并开平方根
    return distance.min(axis=0)                                   #返回最小距离


def file2mat(test_filename, para_num):
    """
    将表格存入矩阵，test_filename为表格路径，para_num为存入矩阵的列数
    返回目标矩阵，和矩阵每一行数据的类别
    """
    fr = open(test_filename)
    lines = fr.readlines()
    line_nums = len(lines)
    result_mat = np.zeros((line_nums, para_num))                 #创建line_nums行 para_num列的矩阵
    class_label = []
    for i in range(line_nums):
        line = lines[i].strip()
        item_mat = line.split(',')
        result_mat[i, :] = item_mat[0: para_num]
        class_label.append(item_mat[-1])                         #表格中最后一列正常1异常2的分类存入class_label
    fr.close()
    return result_mat, class_label


def roc(data_set):
    normal = 0
    data_set_size = data_set.shape[1]
    roc_rate = np.zeros((2, data_set_size))
    for i in range(data_set_size):
        if data_set[2][i] == 1:
            normal += 1
    abnormal = data_set_size - normal
    max_dis = data_set[1].max()
    for j in range(1000):
        threshold = max_dis / 1000 * j
        normal1 = 0
        abnormal1 = 0
        for k in range(data_set_size):
            if data_set[1][k] > threshold and data_set[2][k] == 1:
                normal1 += 1
            if data_set[1][k] > threshold and data_set[2][k] == 2:
                abnormal1 += 1
        roc_rate[0][j] = normal1 / normal         #阈值以上正常点/全体正常的点
        roc_rate[1][j] = abnormal1 / abnormal     #阈值以上异常点/全体异常点
    return roc_rate


def test(training_filename, test_filename):
    training_mat, training_label = file2mat(training_filename, 32)
    test_mat, test_label = file2mat(test_filename, 32)
    test_size = test_mat.shape[0]
    result = np.zeros((test_size, 3))
    for i in range(test_size):
        result[i] = i + 1, classify(test_mat[i], training_mat), test_label[i]  # 序号 最小欧氏距离 测试集数据类别
    result = np.transpose(result)  #矩阵转置
    plt.figure(1)
    plt.scatter(result[0], result[1], c=result[2], edgecolors='None', s=1, alpha=1)
    # 图1 散点图：横轴为序号，纵轴为最小欧氏距离，点中心颜色根据测试集数据类别而定， 点外围无颜色，点大小为最小1，灰度为最大1
    roc_rate = roc(result)
    plt.figure(2)
    plt.scatter(roc_rate[0], roc_rate[1], edgecolors='None', s=1, alpha=1)
    # 图2 ROC曲线：横轴误报率，即阈值以上正常点/全体正常的点；纵轴检测率，即阈值以上异常点/全体异常点
    plt.show()

if __name__ == "__main__":
    test('training.csv', 'test.csv')
