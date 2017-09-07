from bs4 import BeautifulSoup
import pandas as pd
from Word2VecUtil import Word2VecUtil
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from sklearn.linear_model import LogisticRegression
import os
import nltk
from numpy import *

def get_reviews_vector(model,reviews):
    review_vector=zeros((len(reviews),1000))
    for (i,review) in enumerate(reviews):
        nword=0
        for word in review:
            if word in model:
                review_vector[i,:]=review_vector[i,:]+model[word]
                nword=nword+1
        review_vector[i,:]=review_vector[i,:]/nword
    return review_vector

def get_data(model,train_df,val_df,test_df):
    labeled_reviews=[];val_reviews=[];test_reviews=[]
    for review in train_df['review']:
        labeled_reviews.append(Word2VecUtil.review_to_wordlist(review))
    for review in val_df['review']:
        val_reviews.append(Word2VecUtil.review_to_wordlist(review))
    for review in test_df['review']:
        test_reviews.append(Word2VecUtil.review_to_wordlist(review))
    train_x=get_reviews_vector(model,labeled_reviews)
    train_y=train_df['sentiment'].values
    val_x=get_reviews_vector(model,val_reviews)
    test_x=get_reviews_vector(model,test_reviews)
    return train_x,train_y,val_x,test_x

def get_data_array(model,dataframe):
    data_array=zeros((dataframe.values.shape[0],1000))
    for (i,label_id) in enumerate(dataframe['id'].values):
        data_array[i,:]=model.docvecs[label_id]
    return data_array

def predict_model_proba1(model,train_df,val_df,test_df):
    train_x,train_y,val_x,test_x=get_data(model,train_df,val_df,test_df)
    lr_model=LogisticRegression()
    lr_model.fit(train_x,train_y)
    val_feature=lr_model.predict_proba(val_x)[:,1]
    test_feature=lr_model.predict_proba(test_x)[:,1]
    return val_feature,test_feature

def predict_model_proba2(model,train_df,val_df,test_df):
    train_x=get_data_array(model,train_df)
    train_y=train_df['sentiment'].values
    val_x=get_data_array(model,val_df)
    test_x=get_data_array(model,test_df)
    lr_model=LogisticRegression()
    lr_model.fit(train_x,train_y)
    val_feature=lr_model.predict_proba(val_x)[:,1]
    test_feature=lr_model.predict_proba(test_x)[:,1]
    return val_feature,test_feature

if __name__ == '__main__':
    labeled_df=pd.read_csv("data\\labeledTrainData.tsv",delimiter="\t",quoting=3)
    test_df=pd.read_csv("data\\testData.tsv",delimiter="\t",quoting=3)
    # 数据切分操作
    train_df=labeled_df.iloc[0:20000]
    val_df=labeled_df.iloc[20000:]
    model1=Word2Vec.load("1000features_5minwords_10context")
    model2=Doc2Vec.load("1000features_1minwords_10context_dm")
    model3=Doc2Vec.load("1000features_1minwords_10context_bow")
    val_feature1,test_feature1=predict_model_proba1(model1,train_df,val_df,test_df)
    val_feature2,test_feature2=predict_model_proba2(model2,train_df,val_df,test_df)
    val_feature3,test_feature3=predict_model_proba2(model3,train_df,val_df,test_df)
    val_x=hstack((val_feature1.reshape(-1,1),val_feature2.reshape(-1,1),val_feature3.reshape(-1,1)))
    test_x=hstack((test_feature1.reshape(-1,1),test_feature2.reshape(-1,1),test_feature3.reshape(-1,1)))
    lr_model=LogisticRegression()
    lr_model.fit(val_x,val_df['sentiment'].values)
    pred_y=lr_model.predict(test_x)
    submission=pd.DataFrame({'id':test_df['id'],'sentiment':pred_y})
    submission.to_csv('submission.csv',index=False,quoting=3)