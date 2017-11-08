from bs4 import BeautifulSoup
import pandas as pd
from Word2VecUtil import Word2VecUtil
import os
import numpy as np
import random
import tensorflow as tf

def build_single_cell(num_units,has_res):
    single_cell=tf.contrib.rnn.BasicLSTMCell(num_units)
    single_cell=tf.contrib.rnn.DropoutWrapper(cell=single_cell,input_keep_prob=0.8)
    if has_res:
        single_cell=tf.contrib.rnn.ResidualWrapper(single_cell)
    single_cell=tf.contrib.rnn.DeviceWrapper(single_cell,"/cpu:0")
    return single_cell

def build_rnn_cell(normal_layers,residual_layers,num_units):
    cell_list=[]
    for i in range(normal_layers):
        has_res=(i>=(normal_layers-residual_layers))
        cell=build_single_cell(num_units,has_res)
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
        self.weights=tf.Variable(tf.random_normal([num_units*2,2]))
        self.bias=tf.Variable(tf.random_normal([2]))
        source=tf.transpose(iterator.source)
        length=iterator.source_sequence_length
        target=iterator.source_sentiment
        encoder_emb_inp=tf.nn.embedding_lookup(self.embedding_encoder,source)
        self.encoder_emb_inp=encoder_emb_inp
        normal_layers=int(num_layers/2)
        residual_layers=int((num_layers-1)/2)
        fw_cell=build_rnn_cell(normal_layers,residual_layers,num_units)
        bw_cell=build_rnn_cell(normal_layers,residual_layers,num_units)
        bi_outputs,bi_states=tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,encoder_emb_inp,dtype=tf.float32,sequence_length=length,time_major=True)
        self.bi_outputs=tf.concat(bi_outputs,-1)
        logits=tf.matmul(self.bi_outputs[-1],self.weights)+self.bias
        pred=tf.nn.softmax(logits)
        self.result=pred[:,1]
        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=target))
        optimizer=tf.train.AdamOptimizer()
        self.train_op=optimizer.minimize(self.loss)
        correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(target,1))
        self.accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    def train(self):
        return self.train_op
    def train_result(self):
        return self.train_op,self.accuracy,self.loss