from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

cancer = datasets.load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=100)
clf = svm.SVC(kernel='linear')

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))

weights = clf.coef_[0]
names = cancer.feature_names
labels = cancer.target_names
x_ch = 2
y_ch = 1

data = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns=np.append(cancer['feature_names'], ['target']))
colors = ['blue', 'red']
fig, ax = plt.subplots()
x_max = data[names[x_ch]].max()
x_min = data[names[x_ch]].min()
y_max = data[names[y_ch]].max()
y_min = data[names[y_ch]].min()

xx = np.linspace(x_min, x_max, 100)
a = - weights[x_ch] / weights[y_ch]
yy = a * xx + clf.intercept_[0] / weights[y_ch]

ax.plot(xx, yy, 'k-')
data[data['target'] == 0].plot.scatter(x=names[x_ch], y=names[y_ch], color=colors[0], ax=ax, label=labels[0])
data[data['target'] == 1].plot.scatter(x=names[x_ch], y=names[y_ch], color=colors[1], ax=ax, label=labels[1])
plt.ylim([y_min, y_max])
plt.xlim([x_min, x_max])
plt.show()