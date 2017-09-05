from bs4 import BeautifulSoup
import pandas as pd
from Word2VecUtil import Word2VecUtil
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import os
import nltk

def getCleanLabeledReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append(Word2VecUtil.review_to_wordlist(review))
    labelized = []
    for i, id_label in enumerate(reviews["id"]):
        labelized.append(LabeledSentence(clean_reviews[i], [id_label]))
        print(labelized)
        os.system('pause')
    return labelized

if __name__ == '__main__':
    train  = pd.read_csv('data\\labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
    train_reviews = getCleanLabeledReviews(train)