import numpy as np
import pydotplus
from sklearn import tree
from sklearn import metrics
from sklearn.externals.six import StringIO
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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

    return feature_train, target_train, feature_test, target_test


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
    target_name = ['didntLike', 'smallDoses', 'largeDoses']

    match_count = 0
    for i in range(len(target_predict)):
        print("Predict result: [%s], Real result: [%s]" % (int(target_predict[i]), int(target_test[i])))
        if target_predict[i] == target_test[i]:
            match_count += 1

    accuracy = float(match_count / len(target_predict))
    print("The accuracy rate is: %.2f%%" % (100 * accuracy))

    result = metrics.classification_report(target_test, target_predict, target_names=target_name)
    print(result)

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
    main()