from bs4 import BeautifulSoup
import pandas as pd
from Word2VecUtil import Word2VecUtil
import os
import numpy as np
import random
import tensorflow as tf

def build_single_cell(num_units):
    single_cell=tf.contrib.rnn.BasicLSTMCell(num_units)
    single_cell=tf.contrib.rnn.DropoutWrapper(cell=single_cell,input_keep_prob=0.8)
    single_cell=tf.contrib.rnn.ResidualWrapper(single_cell)
    single_cell=tf.contrib.rnn.DeviceWrapper(single_cell,"/cpu:0")
    return single_cell

def build_rnn_cell(num_layers,num_units):
    cell_list=[]
    for i in range(num_layers):
        cell=build_single_cell(num_units)
        cell_list.append(cell)
    return tf.contrib.rnn.MultiRNNCell(cell_list)

class RNNEncoder(object):
    def __init__(self,iterator,num_layers,num_units,src_vocab_table):
        self.iterator=iterator
        self.src_vocab_table=src_vocab_table
        embedding_encoder=tf.get_variable('embedding_encoder',shape=[160277,num_units],dtype=tf.float32)
        self.embedding_encoder=embedding_encoder
        self.num_layers=num_layers
        self.num_units=num_units
        source=iterator.source
        length=iterator.source_sequence_length
        encoder_emb_inp=tf.nn.embedding_lookup(self.embedding_encoder,source)
        self.encoder_emb_inp=encoder_emb_inp
        fw_cell=build_rnn_cell(self.num_layers,self.num_units)
        bw_cell=build_rnn_cell(self.num_layers,self.num_units)
        bi_outputs,bi_states=tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,encoder_emb_inp,dtype=tf.float32,sequence_length=length,time_major=True)
        self.bi_outputs=tf.concat(bi_outputs,-1)
    def get_encoder_outputs(self):
        return self.encoder_emb_inp