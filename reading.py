import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


