import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm

from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import RMSprop, Adam
from keras.utils import to_categorical

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score
from sklearn.svm import SVC, LinearSVC

import matplotlib.pyplot as plt
plt.style.use('bmh')

df = pd.read_csv("./Dataset/unb-ids-2018/dataset.csv")
X = df.drop(columns=['Label'], axis=1)
y = df['Label']

print(np.shape(X))
print(np.shape(y))
# print(df['Label'].value_counts())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_cat = to_categorical(y_train, num_classes=11)
y_test_cat = to_categorical(y_test, num_classes=11)

print(X_train_scaled)
print("-------------")
print(X_test_scaled)
print("-------------")
print(np.shape(y_train_cat))
print("-------------")
print(y_test_cat)
print("-------------")

def NN_model():
    model = Sequential()
    model.add(Dense(256,activation='relu',input_shape=(X_train_scaled.shape[1],)))
    model.add(Dropout(0.4))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(y_train_cat.shape[1],activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
    model.summary()

    return model

Neural_Network_Model = NN_model()
Neural_Network_Model.fit(X_train_scaled,y_train_cat,epochs=5,verbose=1,batch_size=64)
scores = Neural_Network_Model.evaluate(X_test_scaled,y_test_cat)
print("Accuracy : ",scores[1]*100)