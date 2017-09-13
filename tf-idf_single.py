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
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.grid_search import GridSearchCV
from keras.models import Sequential,Model
from sklearn.model_selection import train_test_split
from scipy import sparse
import h5py
import os
import nltk
from numpy import *

def make_reviews(labeled_df,test_df):
    labeled_reviews=[];test_reviews=[]
    for review in labeled_df['review']:
        labeled_reviews.append(Word2VecUtil.review_to_wordlist(review))
    for review in test_df['review']:
        test_reviews.append(Word2VecUtil.review_to_wordlist(review))
    return labeled_reviews,test_reviews

def get_vector(labeled_reviews,test_reviews):
    vectorizer=TfidfVectorizer(min_df=3,max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1,2), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words = 'english')
    labeled_string,test_string=[],[]
    for review in labeled_reviews:
        labeled_string.append(" ".join(review))
    for review in test_reviews:
        test_string.append(" ".join(review))
    all_string=labeled_string+test_string
    vectorizer.fit(all_string)
    train_tfidf_x=vectorizer.transform(labeled_string)
    test_tfidf_x=vectorizer.transform(test_string)
    return train_tfidf_x,test_tfidf_x

def get_sgd_model():
    sgd_params={'alpha':[0.00006,0.00007,0.00008,0.0001,0.0005]}
    model_SGD=GridSearchCV(SGD(random_state=0,shuffle=True,loss='modified_huber'),sgd_params,scoring='roc_auc',cv=20)
    return model_SGD

if __name__ == '__main__':
    labeled_df=pd.read_csv("data\\labeledTrainData.tsv",delimiter="\t",quoting=3)
    test_df=pd.read_csv("data\\testData.tsv",delimiter="\t",quoting=3)
    labeled_reviews,test_reviews=make_reviews(labeled_df,test_df)
    train_x,test_x=get_vector(labeled_reviews,test_reviews)
    train_y=labeled_df['sentiment'].values
    print(shape(train_x))
    sgd_model=get_sgd_model()
    sgd_model.fit(train_x,train_y)
    pred_y=sgd_model.predict_proba(test_x)[:,1]
    submission=pd.DataFrame({'id':test_df['id'],'sentiment':pred_y})
    submission.to_csv('submission.csv',index=False,quoting=3)