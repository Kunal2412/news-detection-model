# -*- coding: utf-8 -*-
"""
Name     : Kunal Sachdev
Email id : kunalsachdev456@gmail.com
"""
#import necessary packages/libraries required
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

#changing the default directory
os.chdir("C:/Users/HP/Desktop/Internship(mini-project)")

#reading news.csv file in dataframe df
df=pd.read_csv("news.csv",sep=",")

#renaming the label of first column of dataset
df.rename(columns={'Unnamed: 0':'id'}, inplace=True)

#printing information of dataframe df
print(df.info())

#storing lablel column (i.e. FAKE or REAL ) in y variable
y=df['label']

#splitting dataset into testing and training datasets
xTrain,xTest,yTrain,yTest=train_test_split(df['text'], y, test_size=0.3, random_state=7)

#initialising a TfidfVectorizer filtering stop words from the English language and a maximum document frequency of 0.7
tfidfVectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

# fitting and transforming the vectorizer on the train set, and transforming the vectorizer on the test set.
tfidfTrain=tfidfVectorizer.fit_transform(xTrain) 
tfidfTest=tfidfVectorizer.transform(xTest)

#initialising a passive aggressive classifier 
pac=PassiveAggressiveClassifier(max_iter=50)

#fitting the training dataset to the pac classifier
pac.fit(tfidfTrain,yTrain)

#predicting label(FAKE or REAL) for testing dataset and finding the accuracy
yPred=pac.predict(tfidfTest)
score=accuracy_score(yTest,yPred)
print(f'Accuracy: {round(score*100,2)}%')

#printing the confusion matrix
confusion_matrix(yTest,yPred, labels=['FAKE','REAL'])