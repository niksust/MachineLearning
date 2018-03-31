from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv('all_data.csv')
print data.head()

y = data['label']
X = data.drop('label', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# print(len(y_test))

clf = MLPClassifier(solver='sgd', alpha=1e-5,
                    hidden_layer_sizes=(20,10), random_state=1)

clf.fit(X_train, y_train)
acc=clf.score(X_test,y_test)
# print(acc)
print("Accuracy: %.2f" % (acc*100))

import numpy as np
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, clf.predict(X_test)))

# accuracy_score(y_true, y_pred, normalize=False)



# print(X_test)
# list=[]
# list2=[]
# for i in X_test:
# 	list.append(X_test[i])
	
# for j in y_test:
# 	list2.append(j)

# print(clf.predict(list[0]))


# print(clf.predict(X_test))
# print(y_test)
# predicted=clf.predict(X_test)


