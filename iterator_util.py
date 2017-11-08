from __future__ import print_function
import collections
import tensorflow as tf

__all__=['BatchedInput','get_iterator']

class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer","source","source_sentiment",
                            "source_sequence_length"))):
    pass

def one_hot_label(y):
    trans=tf.cond(tf.equal(y,tf.constant(0)),lambda: tf.constant([1,0]),lambda: tf.constant([0,1]))
    return trans

def get_iterator(src_dataset,src_vocab_table,batch_size):
    src_eos_id=tf.cast(src_vocab_table.lookup(tf.constant('<eos>')),tf.int32)
    src_dataset=src_dataset.map(lambda x,y:(tf.string_split([x]).values,y))
    src_dataset=src_dataset.map(lambda x,y:(tf.cast(src_vocab_table.lookup(x),tf.int32),y))
    src_dataset=src_dataset.map(lambda x,y:(tf.reverse(x,axis=[0]),y))
    src_dataset=src_dataset.map(lambda x,y:(x,tf.size(x),one_hot_label(y)))
    def batching_func(x):    
        return x.padded_batch(batch_size,padded_shapes=(tf.TensorShape([None]),tf.TensorShape([]),tf.TensorShape([2])),padding_values=(src_eos_id,0,0))
    batched_dataset=batching_func(src_dataset)
    batched_iter=batched_dataset.make_initializable_iterator()
    (src_ids,src_seq_len,src_sentiment)=batched_iter.get_next()
    return BatchedInput(initializer=batched_iter.initializer,
        source=src_ids,
        source_sentiment=src_sentiment,
        source_sequence_length=src_seq_len)