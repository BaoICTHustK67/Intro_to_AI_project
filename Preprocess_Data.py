import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import nltk
import string
from tkinter import *
import tkinter as tk


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from nltk.stem.porter import PorterStemmer
import pickle
filename = 'finalized_model.sav'

ps = PorterStemmer()
stopWord = set(stopwords.words('english'))

df = pd.read_csv('dataset/spam_HH.csv', encoding='latin-1')


#Assign Category from [ham, spam] to [0, 1]
encoder = LabelEncoder()
df['Category'] = encoder.fit_transform(df['Category'])
df.drop_duplicates(keep='first')

#Preprocessing Phase
def get_importantFeature(sent):
    sent = sent.lower()

    returnList = []
    sent = nltk.word_tokenize(sent)
    for i in sent:
        if i.isalnum():
            returnList.append(i)
    return returnList 


def removing_stopWord(sent):
    returnList = []
    for i in sent:
        if i not in stopWord and i not in string.punctuation:
            returnList.append(i)
    return returnList


def potter_stem(sent):
    returnList = []
    for i in sent:
        returnList.append(ps.stem(i))
    return " ".join(returnList)


#Apply in 'Message' column
df['imp_Feature'] = df['Message'].apply(get_importantFeature)

df['imp_Feature'] = df['imp_Feature'].apply(removing_stopWord)

df['imp_Feature'] = df['imp_Feature'].apply(potter_stem)

#Split training and test data
X = df['imp_Feature']
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05,
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
pickle.dump(model, open(filename, 'wb'))
spam_model = pickle.load(open("finalized_model.sav",'rb'))
 
def check_spam():
    text = spam_text_Entry.get()
    is_spam = spam_model.predict(tfidf_vectorizer.transform([text]))
    if is_spam == 1:
        print("text is spam")
        my_string_var.set("Result: text is spam")
    else:
        print("text is not spam")
        my_string_var.set("Result: text is not spam")
win = Tk()
 
win.geometry("400x600")
win.configure(background="white")
win.title("Message/Comment Spam Detector")
 
title = Label(win, text="Message/Comment Spam Detector", bg="gray",
              width="300",height="2",fg="white",
              font=("Calibri 20 bold italic underline")).pack()
 
spam_text = Label(win, text="Enter your Text: ",bg="cyan",
                   font=("Verdana 12")).place(x=12,y=100)
spam_text_Entry = Entry(win, textvariable=spam_text,width=33)
spam_text_Entry.place(x=155, y=105)
 
my_string_var = StringVar()
my_string_var.set("Result: ")
 
print_spam = Label(win, textvariable=my_string_var,bg="cyan",
                    font=("Verdana 12")).place(x=12,y=200)
 
Button = Button(win, text="Submit",width="12",height="1",
                activebackground="red",bg="Pink",command=check_spam,
                font=("Verdana 12")).place(x=12,y=150)
 
win.mainloop()







