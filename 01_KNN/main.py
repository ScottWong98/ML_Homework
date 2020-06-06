import csv
import random
import math
import numpy as np
import matplotlib.pyplot as plt


def load_dataset(filename):
    """
    加载并预处理数据集
    将数据集进行随机处理
    :param filename: 数据集文件名
    :return:
        - feature_mat: 特征二维数组
        - class_label_list: 分类标签列表
    """
    # 读取数据集文件
    with open(filename, 'r') as file:
        lines = file.readlines()
    # 去除数据集文件中的标题
    lines = lines[1:]
    random.shuffle(lines)
    # 数据集样本大小
    num_of_lines = len(lines)
    # 初始化特征二维数组(150 * 4)
    feature_mat = np.zeros((num_of_lines, lines[0].count(',') - 1))
    # 初始化分类标签列表
    class_label_list = []
    # 遍历数据集，将数据集转化成特征二维数组和分类标签列表
    for index, line in enumerate(lines):
        list_from_line = line.strip().split(',')
        feature_mat[index, :] = list_from_line[1:5]
        class_label_list.append(list_from_line[-1].replace('\n', '').replace('\r', ''))
    return feature_mat, class_label_list


def split_train_test(feature_mat, class_label_list, test_size=0.25):
    """
    将数据集按比例分成训练集和测试集
    :param feature_mat: 特征二维数组
    :param class_label_list: 分类标签列表
    :param test_size: 测试集所占比例
    :return:
        - feature_training_dataset: 特征训练集
        - label_training_dataset: 标签训练集
        - feature_test_dataset: 特征测试集
        - label_test_dataset: 标签测试集
    """
    # 数据集大小
    data_size = feature_mat.shape[0]
    # 分割比例
    split = math.floor(data_size * test_size)
    # 测试集数据
    feature_test_dataset = feature_mat[0: split]
    label_test_dataset = class_label_list[0: split]
    # 训练集数据
    feature_training_dataset = feature_mat[split:]
    label_training_dataset = class_label_list[split:]
    return feature_training_dataset, label_training_dataset, feature_test_dataset, label_test_dataset


def choose2feature(dataset):
    """
    从数据集中选择两个特征
    :param dataset: 数据集
    :return: 有两个特征的数据集
    """
    return dataset[:, 0:2]


def knn_classifier(test_data, training_dataset, training_labels, k):
    """
    knn分类实现
    :param test_data: 测试数据
    :param training_dataset: 训练数据集
    :param training_labels: 训练标签集
    :param k: k值
    :return: 预测标签值
    """
    # 训练集大小
    training_dataset_size = training_dataset.shape[0]

    # 计算欧式距离
    diff_mat = np.tile(test_data, (training_dataset_size, 1)) - training_dataset
    sq_diff_mat = diff_mat ** 2
    distances = (sq_diff_mat.sum(axis=1)) ** 0.5

    # 对距离进行从小到大排序, 生成排序后的下标
    sorted_distance_indices = distances.argsort()
    # 取前k个标签类别计数
    class_count = {}

    # 生成class_count: 前k个的标签类别数目
    for i in range(k):
        vote_label = training_labels[sorted_distance_indices[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1

    # 取前k个出现次数最多的标签，即knn的预测标签
    sorted_class_count = sorted(class_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_class_count[0][0]


def auto_norm(dataset):
    """
    规范化数据
    :param dataset: 数据集
    :return: 规范化的数据集
    """
    min_val = np.min(dataset, axis=0)
    max_val = np.max(dataset, axis=0)
    ranges = max_val - min_val
    m = dataset.shape[0]
    norm_dataset = (dataset - np.tile(min_val, (m, 1))) / ranges
    return norm_dataset


def iris_class_test(k):
    """
    Iris分类测试
    """
    # 预处理数据
    feature_mat, class_label_vector = load_dataset('iris.csv')
    # 获取测试集和训练集
    feature_training_dataset, label_training_dataset, feature_test_dataset, label_test_dataset = split_train_test(
        feature_mat, class_label_vector, 0.3)

    # 选择两个属性进行k值分类
    # feature_training_dataset = choose2feature(feature_training_dataset)
    # feature_test_dataset = choose2feature(feature_test_dataset)

    # 对数据集进行规范化
    feature_training_dataset = auto_norm(feature_training_dataset)
    feature_test_dataset = auto_norm(feature_test_dataset)
    # 测试集大小
    test_size = feature_test_dataset.shape[0]
    # 设置k取值
    # k = 2
    # 预测成功个数
    success = 0
    # 预测标签列表
    label_predict_dataset = []
    for i in range(test_size):
        predict_label = knn_classifier(feature_test_dataset[i, :], feature_training_dataset, label_training_dataset, k)
        label_predict_dataset.append(predict_label)
        # print("predict label: [%s], real label: [%s]" % (predict_label, label_test_dataset[i]))
        if predict_label == label_test_dataset[i]:
            success += 1
    #plot_figure(feature_test_dataset, label_predict_dataset, label_test_dataset)

    print("The accuracy rate is: %.2f%%" % (100 * success / float(test_size)))


def plot_figure(dataset, predict_labels, real_labels):
    """
    画出预测结果和实际结果的散点图
    :param dataset: 数据集
    :param predict_labels: 预测结果
    :param real_labels: 实际结果
    """
    # iris名字
    iris_name = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    labels = np.array(predict_labels)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.scatter(dataset[labels == iris_name[0], 0], dataset[labels == iris_name[0], 1], color='g', label=iris_name[0])
    ax1.scatter(dataset[labels == iris_name[1], 0], dataset[labels == iris_name[1], 1], color='r', label=iris_name[1])
    ax1.scatter(dataset[labels == iris_name[2], 0], dataset[labels == iris_name[2], 1], color='b', label=iris_name[2])
    plt.xlabel("sepal length")
    plt.ylabel("sepal width")
    plt.title("Predict Result")
    labels = np.array(real_labels)
    ax2 = fig.add_subplot(122)
    ax2.scatter(dataset[labels == iris_name[0], 0], dataset[labels == iris_name[0], 1], color='g', label=iris_name[0])
    ax2.scatter(dataset[labels == iris_name[1], 0], dataset[labels == iris_name[1], 1], color='r', label=iris_name[1])
    ax2.scatter(dataset[labels == iris_name[2], 0], dataset[labels == iris_name[2], 1], color='b', label=iris_name[2])
    plt.xlabel("sepal length")
    plt.ylabel("sepal width")
    plt.title("Real Result")
    plt.legend()
    plt.show()


def main():
    """
    主函数
    """
    # iris_class_test(1)
    # iris_class_test(2)
    # iris_class_test(3)
    # iris_class_test(4)
    iris_class_test(5)
    # iris_class_test(6)
    # iris_class_test(7)


if __name__ == '__main__':
    main()
