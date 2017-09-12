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

def get_data_array(model,dataframe):
    data_array=zeros((dataframe.values.shape[0],5000))
    for (i,label_id) in enumerate(dataframe['id'].values):
        data_array[i,:]=model.docvecs[label_id]
    return data_array

if __name__ == '__main__':
    labeled_df=pd.read_csv("data\\labeledTrainData.tsv",delimiter="\t",quoting=3)
    test_df=pd.read_csv("data\\testData.tsv",delimiter="\t",quoting=3)
    model_dm_name="5000features_1minwords_10context_dm"
    model_bow_name="5000features_1minwords_10context_bow"
    model_dm=Doc2Vec.load(model_dm_name)
    model_bow=Doc2Vec.load(model_bow_name)
    dm_train_x=get_data_array(model_dm,labeled_df)
    bow_train_x=get_data_array(model_bow,labeled_df)
    dm_test_x=get_data_array(model_dm,test_df)
    bow_test_x=get_data_array(model_bow,test_df)
    train_x=hstack((dm_train_x,bow_train_x))
    train_y=labeled_df['sentiment'].values
    test_x=hstack((dm_test_x,bow_test_x))
    lr_model=LogisticRegression()
    lr_model.fit(train_x,train_y)
    pred_y=lr_model.predict(test_x)
    submission=pd.DataFrame({'id':test_df['id'],'sentiment':pred_y})
    submission.to_csv('submission.csv',index=False,quoting=3)