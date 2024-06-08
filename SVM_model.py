import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import GUI
import pickle
from Preprocessing import *

from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import datasets

filename = 'finalized_model.sav'
df1 = pd.read_csv('dataset\spam1.csv', encoding='latin-1')
df2 = pd.read_csv('dataset\spam2.csv', encoding='latin-1')

df1.v1.value_counts().plot(kind='bar')

df1.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1, inplace=True)

df1.columns = ['label', 'Message']

df2 = df2[df2['labels'] == 'spam']
df2 = df2[df2['lang'] == 'english']
df2.drop(['lang', 'sentiment'], axis=1, inplace=True)
df2.columns = ['label', 'Message']

df = pd.concat([df1, df2], axis=0)

df['Category'] = df['label'].map({'ham' : 0, 'spam' : 1})
df.drop(['label'], axis=1, inplace=True)
#Preprocessing Phase

#Apply in 'Message' column
df['imp_Feature'] = df['Message'].apply(get_importantFeature)

df['imp_Feature'] = df['imp_Feature'].apply(removing_stopWord)

df['imp_Feature'] = df['imp_Feature'].apply(potter_stem)

#Split training and test data
X = df['imp_Feature']
y = df['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2  ,
                                                     random_state=1)


#Fit in svm 
tfidf_vectorizer = TfidfVectorizer()
feature = tfidf_vectorizer.fit_transform(X_train)



tuned_parameters = {'kernel':['linear','rbf'],'gamma':[1e-3,1e-4], 'C':
                    [1,10,100,1000]}

model = GridSearchCV(svm.SVC(), tuned_parameters, cv=5)
model.fit(feature, y_train)
#Predict 

y_predict = model.predict(tfidf_vectorizer.transform(X_test))


acc_score = accuracy_score(y_predict, y_test)
pren_score = precision_score(y_predict, y_test)
f1 = f1_score(y_predict, y_test)
recall = recall_score(y_predict, y_test)

print("Accuracy: ", acc_score)
print("Precision: ", pren_score)
print("Recall: ", recall)
print("F1 Score: ", f1)

#Checking spam 
GUI.Build_GUI(model, tfidf_vectorizer, filename)