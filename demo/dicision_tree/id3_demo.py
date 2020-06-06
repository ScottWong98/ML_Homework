from math import log
import operator
import numpy as np
import pydotplus
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],                        #数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']        #特征标签
    return dataSet, labels                             #返回数据集和分类属性


def load_data(filename):
    """
    加载数据，利用sklearn将数据分成训练集和测试集
    :param filename: 文件名
    :return:
        - feature_training: 特征训练集
        - target_training: 标签训练集
        - feature_test: 特征测试集
        - target_test: 标签测试集`
    """
    dataset = np.loadtxt(filename)
    feature_dataset = dataset[:, 0: 3]
    target_dataset = dataset[:, -1]
    feature_train, feature_test, target_train, target_test = train_test_split(feature_dataset, target_dataset, test_size=0.3)

    test_dataset = np.c_[feature_train, target_train]
    train_dataset = np.c_[feature_test, target_test]

    # for i in range(feature_train.shape[0]):
    #     temp = feature_train[i]
    #     temp = temp.extend(target_train[i])
    #     train_dataset.append(temp)
    # for i in range(feature_test.shape[0]):
    #     temp = feature_test[i]
    #     temp = temp.extend(target_test[i])
    #     test_dataset.append(temp)

    return train_dataset, test_dataset


def calc_shannon_ent(dataset):
    """
    计算香农熵
    :param dataset: 数据集
    :return: 香农熵
    """
    # 数据集大小
    n = len(dataset)
    # 每个标签出现的次数
    label_counts = {}
    for feat_vec  in dataset:
        current_label = feat_vec[-1]
        label_counts[current_label] = label_counts.get(current_label, 0) + 1

    # 计算香农熵
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / n
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def split_dataset(dataset, axis, value):
    """
    按照给定特征划分数据集
    :param dataset:
    :param axis:
    :param value:
    :return:
    """
    ret_dataset = []
    for feat_list in dataset:
        if feat_list[axis] == value:
            reduced_feats = feat_list[:axis]
            reduced_feats.extend(feat_list[axis + 1:])
            ret_dataset.append(reduced_feats)
    return ret_dataset


def choose_best_feature_to_split(dataset):
    # 特征数量
    m = len(dataset[0]) - 1
    # 计算数据集的香农熵
    base_entropy = calc_shannon_ent(dataset)
    # 信息增益
    best_info_gain = 0.0
    # 最优特征的索引值
    best_feature = -1
    for i in range(m):
        feature_list = [example[i] for example in dataset]
        unique_vals = set(feature_list)
        new_entropy = 0.0
        for value in unique_vals:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset) / float(len(dataset))
            new_entropy += prob * calc_shannon_ent(sub_dataset)
        info_gain = base_entropy - new_entropy
        if info_gain > best_feature:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_count(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key= operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(dataset, labels, feat_labels):
    class_list = [example[-1] for example in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(dataset[0]) == 1 or len(labels) == 0:
        return majority_count(class_list)

    best_feat = choose_best_feature_to_split(dataset)
    best_feat_label = labels[best_feat]
    feat_labels.append(best_feat_label)
    my_tree = {best_feat_label: {}}
    del(labels[best_feat])
    feat_values = [example[best_feat] for example in dataset]
    unique_vals = set(feat_values)
    for value in unique_vals:
        my_tree[best_feat_label][value] = create_tree(split_dataset(dataset, best_feat, value), labels, feat_labels)
    return my_tree


def classify(input_tree, feat_labels, test_vec):
    first_str = next(iter(input_tree))
    print(first_str)
    second_dict = input_tree[first_str]
    print(second_dict)
    feat_index = feat_labels.index(first_str)
    print(feat_index)
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label



def decision_classifier(feature_train, target_train):
    """
    利用sklean实现决策树
    :param feature_train: 特征训练集
    :param target_train: 标签训练集
    :return: 决策树
    """
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(feature_train, target_train)
    return clf


def dating_class_predict():
    """
    测试算法
    """
    feature_train, target_train, feature_test, target_test = load_data('dating_set.txt')
    clf = decision_classifier(feature_train, target_train)

    target_predict = clf.predict(feature_test)
    match_count = 0
    for i in range(len(target_predict)):
        print("Predict result: [%s], Real result: [%s]" % (int(target_predict[i]), int(target_test[i])))
        if target_predict[i] == target_test[i]:
            match_count += 1

    accuracy = float(match_count / len(target_predict))
    print("The accuracy rate is: %.2f%%" % (100 * accuracy))
    plot_figure(feature_test, target_predict, target_test)
    export_tree(clf)


def plot_figure(dataset, predict_labels, real_labels):
    """
    画出预测结果和实际结果的散点图
    :param dataset: 数据集
    :param predict_labels: 预测结果
    :param real_labels: 实际结果
    """
    target_name = ['didntLike', 'smallDoses', 'largeDoses']
    labels = np.array(predict_labels)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.scatter(dataset[labels == 1, 0], dataset[labels == 1, 1], color='g', label=target_name[0])
    ax1.scatter(dataset[labels == 2, 0], dataset[labels == 2, 1], color='r', label=target_name[1])
    ax1.scatter(dataset[labels == 3, 0], dataset[labels == 3, 1], color='b', label=target_name[2])
    plt.xlabel("flymiles")
    plt.ylabel("videogame")
    plt.title("Predict Result")
    labels = np.array(real_labels)
    ax2 = fig.add_subplot(122)
    ax2.scatter(dataset[labels == 1, 0], dataset[labels == 1, 1], color='g', label=target_name[0])
    ax2.scatter(dataset[labels == 2, 0], dataset[labels == 2, 1], color='r', label=target_name[1])
    ax2.scatter(dataset[labels == 3, 0], dataset[labels == 3, 1], color='b', label=target_name[2])
    plt.xlabel("flymiles")
    plt.ylabel("videogame")
    plt.title("Real Result")
    plt.legend()
    plt.show()


def export_tree(clf):
    """
    导出决策树结构
    :param clf:
    :return:
    """
    feature_name = ['flymiles', 'videogame', 'icecream']
    target_name = ['didntLike', 'smallDoses', 'largeDoses']
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data, feature_names=feature_name,
                         class_names=target_name, filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dating_cart.pdf")


def main():
    """
    主函数
    """
    dating_class_predict()


if __name__ == '__main__':
    # main()
    dataset, labels = createDataSet()
    feat_labels = []
    my_tree = create_tree(dataset, labels, feat_labels)
    testVec = [0, 1]
    result = classify(my_tree, feat_labels, testVec)
    print(result)
    # train_dataset, test_dataset = load_data('dating_set.txt')
    # feat_labels = []
    # labels = ['flymiles', 'videogame', 'icecream', 'like']
    # my_tree = create_tree(train_dataset.tolist(), labels, feat_labels)
    # testVec = test_dataset[0]
    # result = classify(my_tree, feat_labels, testVec)
    # print(result)
