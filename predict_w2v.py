from bs4 import BeautifulSoup
import pandas as pd
from Word2VecUtil import Word2VecUtil
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from numpy import *
import os
import nltk

def get_reviews_vector(model,reviews):
    review_vector=zeros((len(reviews),5000))
    for (i,review) in enumerate(reviews):
        nword=0
        for word in review:
            if word in model:
                review_vector[i,:]=review_vector[i,:]+model[word]
                nword=nword+1
        review_vector[i,:]=review_vector[i,:]/nword
    return review_vector

def get_data(model,labeled_df,test_df):
    labeled_reviews=[];test_reviews=[]
    for review in labeled_df['review']:
        labeled_reviews.append(Word2VecUtil.review_to_wordlist(review))
    for review in test_df['review']:
        test_reviews.append(Word2VecUtil.review_to_wordlist(review))
    train_x=get_reviews_vector(model,labeled_reviews)
    train_y=labeled_df['sentiment'].values
    test_x=get_reviews_vector(model,test_reviews)
    return train_x,train_y,test_x

if __name__ == '__main__':
    model=Word2Vec.load("5000features_5minwords_10context")
    labeled_df=pd.read_csv("data\\labeledTrainData.tsv",delimiter="\t",quoting=3)
    test_df=pd.read_csv("data\\testData.tsv",delimiter="\t",quoting=3)
    train_x,train_y,test_x=get_data(model,labeled_df,test_df)
    lr_model=LogisticRegression()
    lr_model.fit(train_x,train_y)
    pred_y=lr_model.predict(test_x)
    print(pred_y[0])
    print(lr_model.predict_proba(test_x)[0])
    # submission=pd.DataFrame({'id':test_df['id'],'sentiment':pred_y})
    # submission.to_csv('submission.csv',index=False,quoting=3)