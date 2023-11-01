from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

irisData = load_iris()
X = irisData.data
y = irisData.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Classes: {}'.format(knn.classes_))

predictions = knn.predict(X_test)
for i in range(len(X_test)):
    print('The data {} is predicted to be in class {}'.format(X_test[i], predictions[i]))

data = pd.DataFrame(data=np.c_[X_test, predictions], columns=irisData['feature_names'] + ['prediction'])
fig, ax = plt.subplots()

x_ch = 0
y_ch = 2
names = irisData['feature_names']
labels = irisData['target_names']
colors = ['red', 'green', 'blue']
for i, p in enumerate(set(predictions)):
    data[data['prediction'] == p].plot.scatter(x=names[x_ch], y=names[y_ch], color=colors[i], ax=ax, label=labels[p])

ax.legend()
plt.show()
