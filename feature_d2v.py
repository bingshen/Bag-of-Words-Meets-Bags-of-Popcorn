from bs4 import BeautifulSoup
import pandas as pd
from Word2VecUtil import Word2VecUtil
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import os
import nltk
from numpy import *

def get_label_sentence(dataframe):
    label_sentence=[]
    for [review,label_id] in dataframe[['review','id']].values:
        wordlist=Word2Vec.review_to_wordlist(review)
        label_sentence.append(LabeledSentence(words=wordlist,tags=[label_id]))
    return label_sentence

if __name__ == '__main__':
    labeled_df=pd.read_csv("data\\labeledTrainData.tsv",delimiter="\t",quoting=3)
    unlabeled_df=pd.read_csv("data\\unlabeledTrainData.tsv",delimiter="\t",quoting=3)
    test_df=pd.read_csv("data\\testData.tsv",delimiter="\t",quoting=3)
    model_dm="1000features_1minwords_10context_dm"
    model_bow="1000features_1minwords_10context_bow"
    labeled_sentence=get_label_sentence(labeled_df)
    unlabeled_sentence=get_label_sentence(unlabeled_df)
    test_sentence=get_label_sentence(test_df)
    total_train=vstack((labeled_sentence,unlabeled_sentence,test_sentence))