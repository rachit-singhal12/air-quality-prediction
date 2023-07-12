#importing libraries
import numpy as np
import pandas as pd

 #import training data
data = pd.read_csv('Train.csv')
x_train = data.iloc[:,0:5]
y_train = data.iloc[:,5]

#import testing data
test_data = pd.read_csv('Test.csv')

#train model SVR
from sklearn.svm import SVR
model = SVR(kernel="rbf")

#fit the training data in the model
model.fit(x_train,y_train)

#predict the result using testing data from the model
values = model.predict(test_data)

from sklearn import metrics
pred = model.predict(x_train)
score1 = metrics.r2_score(y_train,pred)
score2 = metrics.mean_absolute_error(y_train,pred)

print(score1,score2)

import pickle as pkl

pkl.dump(model,open('models.pkl','wb'))

testing = pkl.load(open('models.pkl','rb'))

v = testing.predict([[1,2,1,2,1]])
print(v)