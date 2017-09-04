from bs4 import BeautifulSoup
import pandas as pd
from Word2VecUtil import Word2VecUtil
from gensim.models import Word2Vec
import os
import nltk

if __name__ == '__main__':
    labeled_df=pd.read_csv("data\\labeledTrainData.tsv",delimiter="\t",quoting=3)
    unlabeled_df=pd.read_csv("data\\unlabeledTrainData.tsv",delimiter="\t",quoting=3)
    test_df=pd.read_csv("data\\testData.tsv",delimiter="\t",quoting=3)
    model_name="1000features_5minwords_10context";sentences=[]
    for review in labeled_df['review']:
        sentences.append(Word2VecUtil.review_to_wordlist(review))
    for review in unlabeled_df['review']:
        sentences.append(Word2VecUtil.review_to_wordlist(review))
    for review in test_df['review']:
        sentences.append(Word2VecUtil.review_to_wordlist(review))
    model=Word2Vec(sentences,workers=8,size=1000,min_count=5,window=10)
    model.save(model_name)