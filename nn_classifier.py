import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn
import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import warnings
import matplotlib.pyplot as plt

heart_df = pd.read_csv('heart.csv')

#Check that the CSV was imported as a DataFrame
#print(heart_df.head())

#https://www.kaggle.com/rajeshjnv/heart-disease-classification-neural-network

#Get dummy variables for Chest pain (cp)
chest_pain = pd.get_dummies(heart_df['cp'], prefix='cp', drop_first=True)
heart_df = pd.concat([heart_df, chest_pain], axis=1)
heart_df.drop(['cp'], axis=1, inplace=True)

#Get dummy variables for slope (slope)
slp = pd.get_dummies(heart_df['slope'], prefix='slope')

#Get dummy variables for thal (thal)
thal = pd.get_dummies(heart_df['thal'], prefix='thal')

heart_df = pd.concat([heart_df, slp, thal], axis=1)
heart_df.drop(['slope', 'thal'], axis=1, inplace=True)

#print(heart_df.head())

X = heart_df.drop(['target'], axis=1)
y = heart_df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=47)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

network = Sequential()

network.add(Dense(output_dim = 50, init = 'uniform', activation = 'relu', input_dim = 20))

network.add(Dense(output_dim = 50, init = 'uniform', activation = 'relu'))

network.add(Dense(output_dim = 1, init = 'uniform', activation = 'softmax'))

network.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

network.fit(X_train, y_train, batch_size = 10, nb_epoch = 200)

y_pred = network.predict(X_test)

import seaborn as sns
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred.round())
sns.heatmap(cm,annot=True,cmap="Blues",fmt="d",cbar=False)

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred.round())
print('accuracy of the model: ',ac)

plt.show()
