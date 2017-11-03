import tensorflow as tf
from tensorflow.contrib.data import Dataset
import pandas as pd
from iterator_util import *
from tensorflow.python.ops import lookup_ops
from Word2VecUtil import Word2VecUtil
from encoder_model import RNNEncoder
import numpy as np

def load_data(dataframe):
    train_x,train_y=[],[]
    for [review,sentiment] in dataframe[['review','sentiment']].values:
        wordlist=Word2VecUtil.review_to_wordlist(review)
        train_x.append(' '.join(wordlist))
        train_y.append(sentiment)
    return train_x,train_y

if __name__ == '__main__':
    train_df=pd.read_csv("data/labeledTrainData.tsv",delimiter="\t",quoting=3)
    src_vocab_table=lookup_ops.index_table_from_file('data/vocab.txt',default_value=0)
    train_x,train_y=load_data(train_df)
    src_dataset=tf.contrib.data.Dataset.from_tensor_slices((train_x,train_y))
    iterator=get_iterator(src_dataset,src_vocab_table,2,500)
    encoder=RNNEncoder(iterator,2,100,src_vocab_table)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        sess.run(iterator.initializer)
        em=sess.run(encoder.get_encoder_outputs())
        print(np.shape(em))
        em=sess.run(encoder.get_encoder_outputs())
        print(np.shape(em))
        em=sess.run(encoder.get_encoder_outputs())
        print(np.shape(em))