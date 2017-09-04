from bs4 import BeautifulSoup
import pandas as pd
from Word2VecUtil import Word2VecUtil
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from numpy import *
import os
import nltk

def get_reviews_vector(model,reviews):
    review_vector=zeros((len(reviews),1000))
    for (i,review) in enumerate(reveiws):
        nword=0
        for word in review:
            review_vector[i]=review_vector[i]+model[word]
            nword=nword+1
        review_vector=review_vector/nword
    return review_vector

def get_data(model,labeled_df,test_df):
    labeled_reviews=[];test_reviews=[]
    for review in labeled_df['review']:
        labeled_reviews+=Word2VecUtil.review_to_wordlist(review)
    for review in test_df['review']:
        test_reviews+=Word2VecUtil.review_to_wordlist(review)
    train_x=get_reviews_vector(labeled_reviews)
    train_y=labeled_df['sentiment']
    test_x=get_reviews_vector(test_reviews)
    return train_x,train_y,test_x

if __name__ == '__main__':
    model=Word2Vec.load("1000features_5minwords_10context")
    labeled_df=pd.read_csv("data\\labeledTrainData.tsv",delimiter="\t",quoting=3)
    test_df=pd.read_csv("data\\testData.tsv",delimiter="\t",quoting=3)
    train_x,train_y,test_x=get_data(model,labeled_df,test_df)
    lr_model=LogisticRegression()
    lr_model.fit(train_x,train_y)
    pred_y=lr_model.predict(test_x)
    submission=pd.DataFrame({'id':test_df['id'].values,'sentiment':pred_y})
    submission.to_csv('submission.csv',index=False)