import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier #for the model
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import export_graphviz #plot tree
from sklearn.metrics import roc_curve, auc #for model evaluation
from sklearn.metrics import classification_report #for model evaluation
from sklearn.metrics import confusion_matrix #for model evaluation
from sklearn.model_selection import train_test_split
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

heart_df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']

#Create the categorical data, will be used to one hot encoding later.
heart_df['sex'][heart_df['sex'] == 0] = 'female'
heart_df['sex'][heart_df['sex'] == 1] = 'male'

heart_df['chest_pain_type'][heart_df['chest_pain_type'] == 1] = 'typical angina'
heart_df['chest_pain_type'][heart_df['chest_pain_type'] == 2] = 'atypical angina'
heart_df['chest_pain_type'][heart_df['chest_pain_type'] == 3] = 'non-anginal pain'
heart_df['chest_pain_type'][heart_df['chest_pain_type'] == 4] = 'asymptomatic'

heart_df['fasting_blood_sugar'][heart_df['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
heart_df['fasting_blood_sugar'][heart_df['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'

heart_df['rest_ecg'][heart_df['rest_ecg'] == 0] = 'normal'
heart_df['rest_ecg'][heart_df['rest_ecg'] == 1] = 'ST-T wave abnormality'
heart_df['rest_ecg'][heart_df['rest_ecg'] == 2] = 'left ventricular hypertrophy'

heart_df['exercise_induced_angina'][heart_df['exercise_induced_angina'] == 0] = 'no'
heart_df['exercise_induced_angina'][heart_df['exercise_induced_angina'] == 1] = 'yes'

heart_df['st_slope'][heart_df['st_slope'] == 1] = 'upsloping'
heart_df['st_slope'][heart_df['st_slope'] == 2] = 'flat'
heart_df['st_slope'][heart_df['st_slope'] == 3] = 'downsloping'

heart_df['thalassemia'][heart_df['thalassemia'] == 1] = 'normal'
heart_df['thalassemia'][heart_df['thalassemia'] == 2] = 'fixed defect'
heart_df['thalassemia'][heart_df['thalassemia'] == 3] = 'reversable defect'

#Change dtype of the categories to objects.
heart_df['sex'] = heart_df['sex'].astype('object')
heart_df['chest_pain_type'] = heart_df['chest_pain_type'].astype('object')
heart_df['fasting_blood_sugar'] = heart_df['fasting_blood_sugar'].astype('object')
heart_df['rest_ecg'] = heart_df['rest_ecg'].astype('object')
heart_df['exercise_induced_angina'] = heart_df['exercise_induced_angina'].astype('object')
heart_df['st_slope'] = heart_df['st_slope'].astype('object')
heart_df['thalassemia'] = heart_df['thalassemia'].astype('object')

#One hot encode all categorical data.
heart_df = pd.get_dummies(heart_df, drop_first=True)


#Scaling does help classification slightly.
to_be_scaled = ['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate_achieved', 'st_depression', 'num_major_vessels']

for x in to_be_scaled:
    scaled= normalize(heart_df[str(x)].values.reshape(-1,1), axis=0)
    heart_df[str(x)] = scaled

heart_df.to_csv('scaled.csv')

X_train, X_test, y_train, y_test = train_test_split(heart_df.drop('target',1), heart_df['target'], test_size=.2, random_state=10)

model = RandomForestClassifier(max_depth=5, n_estimators=25)
model.fit(X_train, y_train)

y_predict = model.predict(X_test)
y_pred_quant = model.predict_proba(X_test)[:,1]


confusion_matrix = confusion_matrix(y_test, y_predict)
print(confusion_matrix)

conf_m0 = float(confusion_matrix[0,0])
conf_m1 = float(confusion_matrix[1,0])
conf_m2 = float(confusion_matrix[1,1])
conf_m3 = float(confusion_matrix[0,1])

#Good metrics to evaluate are Sensitifity and Specificity
total = sum(sum(confusion_matrix))
sensitivity = conf_m0/(conf_m0 + conf_m1)
print('Sensitivity : ', sensitivity )

specificity = conf_m2/(conf_m2 + conf_m3)
print('Specificity : ', specificity)

#Create a ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)

auc = auc(fpr, tpr)
print(auc)
