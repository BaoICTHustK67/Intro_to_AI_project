import pandas as pd
import numpy as np
import nltk
import GUI
from Preprocessing import *

from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

filename = 'finalized_model'
df = pd.read_csv('dataset/spam.csv', encoding='latin-1')


#Assign Category from [ham, spam] to [0, 1]
encoder = LabelEncoder()
df['Category'] = encoder.fit_transform(df['Category'])
df.drop_duplicates(keep='first')

#Preprocessing Phase

#Apply in 'Message' column
df['imp_Feature'] = df['Message'].apply(get_importantFeature)

df['imp_Feature'] = df['imp_Feature'].apply(removing_stopWord)

df['imp_Feature'] = df['imp_Feature'].apply(potter_stem)

#Split training and test data
X = df['imp_Feature']
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                     random_state=42)

#Fit in svm 
tfidf_vectorizer = TfidfVectorizer()
feature = tfidf_vectorizer.fit_transform(X_train)

tuned_parameters = {'kernel':['linear','rbf'],'gamma':[1e-3,1e-4], 'C':
                    [1,10,100,1000]}

model = GridSearchCV(svm.SVC(), tuned_parameters)
model.fit(feature, y_train)

#Predict 
y_predict = tfidf_vectorizer.transform(X_test)

print("Accuracy: ", model.score(y_predict, y_test)) 

#Checking spam 
GUI.Build_GUI(model, tfidf_vectorizer, filename)