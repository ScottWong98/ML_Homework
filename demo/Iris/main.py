from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
Y = iris.target
# print(X[0:10])
# print(Y[0:10])
# print(iris.DESCR)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

print(X_train)
print(Y_train)
# print(len(X_train))
# print(len(X_test))
# print(X_train[0:10])
# print(Y_train[0:10])
# X_train = X_train[:, 0:2]
# X_test = X_test[:, 0:2]
# print(X_train)

# std = StandardScaler()
# X_train = std.fit_transform(X_train)
# X_test = std.transform(X_test)
# print(X_train[0:10])
#
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train, Y_train)
#
# Y_predict = knn.predict(X_test[0:10])
# print("Real result: ", Y_test[0:10])
# print("Predict result: ", Y_predict)
#
# print("Accuracy: ", knn.score(X_test, Y_test))
plt.scatter(X_train[Y_train==0,0],X_train[Y_train==0,1],color='g',label="symbol 0")
plt.scatter(X_train[Y_train==1,0],X_train[Y_train==1,1],color='r',label="symbol 1")
plt.scatter(X_train[Y_train==2,0],X_train[Y_train==2,1],color='b',label="symbol 2")
plt.title("k-NN view")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.legend()
plt.show()
