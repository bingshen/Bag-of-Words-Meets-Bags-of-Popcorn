from bs4 import BeautifulSoup
import pandas as pd
from Word2VecUtil import Word2VecUtil
import os
import numpy as np
import random
import tensorflow as tf

class BasicLSTM(object):
    def __init__(self,iterator,num_layers,num_units,src_vocab_table,embedding_size):
        self.iterator=iterator
        self.src_vocab_table=src_vocab_table
        self.src_vocab_size=src_vocab_table.size()
        self.embedding_size=embedding_size
        embedding_encoder=tf.get_variable([src_vocab_size,embedding_size],dtype=tf.float32,name='embedding_encoder')
        self.embedding_encoder=embedding_encoder
        self.num_layers=num_layers
        self.num_units=num_units
    def build_encoder(self):
        iterator=self.iterator
        source=iterator.source
        length=iterator.source_sequence_length
        encoder_emb_inp=tf.nn.lookup_embedding(self.embedding_encoder,source)
    def build_single_cell(self):
        num_units=self.num_units
        single_cell=tf.contrib.rnn.BasicLSTMCell(num_units)
        single_cell=tf.contrib.rnn.DropoutWrapper(cell=single_cell,input_keep_prob=0.8)