import time
import pydotplus
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.model_selection import train_test_split


# dot_data = StringIO()
# iris = load_iris()
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(iris.data, iris.target)
# feature_name = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
# target_names = ['Class1', 'Class2', 'Class3']
# tree.export_graphviz(clf, out_file=dot_data, feature_names=feature_name, class_names=target_names, filled=True,
#                      rounded=True, special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("iris.pdf")

data = np.loadtxt('dating_set.txt')
x = data[:, 0 : 3]
y = data[:, -1]
X_train, X_test, Y_train, Y_test= train_test_split(x, y, test_size=0.3)

start_time = time.time()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)

print(time.time() - start_time)

Y_predict = clf.predict(X_test)
match_count = 0
for i in range(len(Y_predict)):
    if Y_predict[i] == Y_test[i]:
        match_count += 1

accuracy = float(match_count / len(Y_predict))

print("Accuray: %.3f%%" % (accuracy * 100))

feature_name = ['flymiles', 'videogame', 'icecream']
target_name = ['didntLike', 'smallDoses', 'largeDoses']

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names = feature_name,
                     class_names=target_name, filled=True, rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("dating.pdf")