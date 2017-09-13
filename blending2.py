from bs4 import BeautifulSoup
import pandas as pd
from Word2VecUtil import Word2VecUtil
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.layers import Dense,Dropout,Activation,Input
from keras.models import Sequential,Model
from sklearn.model_selection import train_test_split
from scipy import sparse
import h5py
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

def get_data_array(model,dataframe):
    data_array=zeros((dataframe.values.shape[0],1000))
    for (i,label_id) in enumerate(dataframe['id'].values):
        data_array[i,:]=model.docvecs[label_id]
    return scale(data_array)

def make_reviews(labeled_df,test_df):
    labeled_reviews=[];test_reviews=[]
    for review in labeled_df['review']:
        labeled_reviews.append(Word2VecUtil.review_to_wordlist(review))
    for review in test_df['review']:
        test_reviews.append(Word2VecUtil.review_to_wordlist(review))
    return labeled_reviews,test_reviews

def get_vector(labeled_reviews,test_reviews):
    vectorizer=TfidfVectorizer(ngram_range=(1,3),sublinear_tf=True)
    labeled_string,test_string=[],[]
    for review in labeled_reviews:
        labeled_string.append(" ".join(review))
    for review in test_reviews:
        test_string.append(" ".join(review))
    vectorizer.fit(labeled_string)
    train_tfidf_x=vectorizer.transform(labeled_string)
    test_tfidf_x=vectorizer.transform(test_string)
    return train_tfidf_x,test_tfidf_x

def one_hot_y(y_label):
    ret=zeros((y_label.shape[0],2))
    for (i,y) in enumerate(y_label):
        ret[i,y]=1
    return ret

def onehot_sample(pred):
    ret=[]
    for [x1,x2] in pred:
        if x1>x2:
            ret.append(0)
        else:
            ret.append(1)
    return ret

def make_deep_model(InputSize):
    deepModel=Sequential()
    deepModel.add(Dense(units=1000,activation='relu',input_shape=(InputSize,)))
    deepModel.add(Dropout(0.5))
    deepModel.add(Dense(units=300,activation='relu'))
    deepModel.add(Dropout(0.2))
    deepModel.add(Dense(units=2,activation="softmax"))
    deepModel.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return deepModel

# 这种融合方法直接让所有的特征拼在一起
if __name__ == '__main__':
    labeled_df=pd.read_csv("data\\labeledTrainData.tsv",delimiter="\t",quoting=3)
    test_df=pd.read_csv("data\\testData.tsv",delimiter="\t",quoting=3)
    model1=Word2Vec.load("1000features_5minwords_10context")
    model2=Doc2Vec.load("1000features_1minwords_10context_dm")
    model3=Doc2Vec.load("1000features_1minwords_10context_bow")
    print("model load over")
    labeled_reviews,test_reviews=make_reviews(labeled_df,test_df)
    train_w2v_x=scale(get_reviews_vector(model1,labeled_reviews))
    test_w2v_x=scale(get_reviews_vector(model1,test_reviews))
    print("w2v load over")
    train_dm_x=get_data_array(model2,labeled_df)
    test_dm_x=get_data_array(model2,test_df)
    print("dm load over")
    train_bow_x=get_data_array(model3,labeled_df)
    test_bow_x=get_data_array(model3,test_df)
    print("bow load over")
    train_tfidf_x,test_tfidf_x=get_vector(labeled_reviews,test_reviews)
    train_x=sparse.hstack((train_w2v_x,train_dm_x,train_bow_x,train_tfidf_x))
    test_x=sparse.hstack((test_w2v_x,test_dm_x,test_bow_x,test_tfidf_x))
    print("tfidf load over")
    train_y=labeled_df['sentiment'].values
    # print(shape(train_x),shape(train_y))
    # fit_x,val_x,fit_y,val_y=train_test_split(train_x,train_y,test_size=0.1,random_state=0)
    # print(shape(fit_x),shape(val_x),shape(fit_y),shape(val_y))
    # deepModel=make_deep_model(train_x.shape[1])
    # deepModel.fit(fit_x.toarray(),fit_y,batch_size=256,epochs=15,verbose=1,validation_data=(val_x.toarray(),val_y),shuffle=True)
    # pred=deepModel.predict(test_x.toarray())
    lr_model=LogisticRegression()
    lr_model.fit(train_x,train_y)
    pred_y=lr_model.predict_proba(test_x)[:,1]
    # with h5py.File("pred2.h5") as h:
    #     h.create_dataset("pred",data=pred_y)
    # print(pred)
    # pred_y=onehot_sample(pred)
    submission=pd.DataFrame({'id':test_df['id'],'sentiment':pred_y})
    submission.to_csv('submission.csv',index=False,quoting=3)