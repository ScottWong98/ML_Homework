import numpy as np
import operator
import matplotlib.pylab as plt


def create_data_set():
    group = np.array([[1., 1.1], [1., 1.], [0., 0.], [0., 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(in_x, data_set, labels, k):
    data_set_size = data_set.shape[0]
    diff_mat = np.tile(in_x, (data_set_size, 1)) - data_set

    sq_diff_mat = diff_mat ** 2
    distance = (sq_diff_mat.sum(axis=1)) ** 0.5

    sorted_distance_indices = distance.argsort()
    class_count = {}

    for i in range(k):
        vote_label = labels[sorted_distance_indices[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1

    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_class_count[0][0]


def file2matrix(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    num_of_lines = len(lines)
    mat = np.zeros((num_of_lines, lines[0].count('\t')))
    class_label_vector = []
    for index, line in enumerate(lines):
        list_from_line = line.strip().split('\t')
        mat[index, :] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))

    return mat, class_label_vector


def plot_figure(data_mat, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    t = ax.scatter(data_mat[:, 1], data_mat[:, 0], 10.0 * np.array(labels), np.array(labels))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title(u'散点图')
    plt.xlabel(u'打机时间')
    plt.ylabel(u'飞机里程')
    plt.show()


def auto_norm(data_set):
    min_val = np.min(data_set, axis=0)
    max_val = np.max(data_set, axis=0)
    ranges = max_val - min_val
    m = data_set.shape[0]
    norm_data_set = (data_set - np.tile(min_val, (m, 1))) / ranges
    return norm_data_set, ranges, min_val


def dating_class_test():
    ratio = 0.1
    mat, labels = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_val = auto_norm(mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * ratio)
    error = 0
    for i in range(num_test_vecs):
        result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :], labels[num_test_vecs:m], 3)
        if result != labels[i]:
            error += 1
    print("the accuracy rate is: %.1f%%" % (100 * (1 - error / float(num_test_vecs))))


if __name__ == '__main__':
    group, labels = create_data_set()
    result = classify0([0, 0], group, labels, 3)
    print('the class of [0, 0] is: ', result)
    mat, labels = file2matrix('datingTestSet2.txt')
    norm_data_set, ranges, min_val = auto_norm(mat)
    plot_figure(norm_data_set, labels)
    dating_class_test()
