from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def report(test_labels, pred_labels):
    true = np.sum(pred_labels == test_labels)
    print("Number of correct prediction:", true)
    print("Number of incorrect prediction:", test_labels.shape[0] - true)
    print("Accuracy:", metrics.accuracy_score(test_labels, pred_labels))
    print(
        "Report of prediction: \n",
        metrics.classification_report(test_labels, pred_labels),
    )


def plot_svc_decision_function(model: svm.SVC):
    ax = plt.gca()

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    ax.contour(
        X, Y, P, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()


def show_plot(X, Y, svc):
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap="rainbow")
    plot_svc_decision_function(svc)


def show(features, labels, svm):
    h = 0.02
    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap="rainbow")
    plt.scatter(
        features[:, 0], features[:, 1], c=labels, cmap="rainbow", edgecolors="k"
    )
    plt.show()


def load_data():
    """ load iris dataset
    return: features, labels
    """
    iris: Any = datasets.load_iris()
    return iris.data, iris.target


def split_data_two():
    """ split data for two-labels classification """
    features, labels = load_data()
    features = features[labels < 2, :2]
    labels = labels[labels < 2]

    return train_test_split(features, labels, test_size=0.2)


def standardization(train_features, test_features):
    """ standardization input features """
    standard_scaler = StandardScaler()
    standard_scaler.fit(train_features)
    train_features = standard_scaler.transform(train_features)
    test_features = standard_scaler.transform(test_features)
    return train_features, test_features


def split_data():
    """ split data for multi-labels classification """
    features, labels = load_data()
    features = features[:, :2]
    return train_test_split(features, labels, test_size=0.2)


def svm_classifier(kernel, train_features, train_labels, test_features, test_labels):
    svc = svm.SVC(kernel=kernel, C=10)
    svc.fit(train_features, train_labels)

    plt.scatter(
        train_features[:, 0], train_features[:, 1], c=train_labels, cmap="rainbow"
    )
    plot_svc_decision_function(svc)

    # Ouput support vector
    print("The support vector is : \n")
    print(svc.support_vectors_)

    pred_labels = svc.predict(test_features)
    report(test_labels, pred_labels)


def svm_multi_classifier(train_features, train_labels, test_features, test_labels):
    """ SVM for multi-labels classifier"""
    svc = svm.SVC(decision_function_shape="ovo")
    svc.fit(train_features, train_labels)
    pred_labels = svc.predict(test_features)
    report(test_labels, pred_labels)
    show(train_features, train_labels, svc)


def main(kernel):
    # Load data for two-labels classification
    train_features, test_features, train_labels, test_labels = split_data_two()

    # Standardization
    train_features, test_features = standardization(train_features, test_features)

    # Two-lables classification
    svm_classifier(kernel, train_features, train_labels, test_features, test_labels)


def multi_main():
    # Load data for multi-labels classification
    train_features, test_features, train_labels, test_labels = split_data()

    # Standardization
    train_features, test_features = standardization(train_features, test_features)

    # Multi-lables classification
    svm_multi_classifier(train_features, train_labels, test_features, test_labels)


if __name__ == "__main__":
    # Uncomment these line for two-labels classification
    # kernels = ['linear', 'poly', 'rbf']
    # for kernel in kernels:
    #     main(kernel)

    # Multi-lables classification
    multi_main()
