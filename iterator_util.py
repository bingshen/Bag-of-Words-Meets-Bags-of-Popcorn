from __future__ import print_function
import collections
import tensorflow as tf

__all__=['BatchedInput','get_iterator']

class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "source", "target_input",
                            "target_output", "source_sequence_length",
                            "target_sequence_length"))):
    pass

def get_iterator(src_dataset,src_vocab_table,batch_size,src_max_len=None):
    src_eos_id=tf.cast(src_vocab_table.lookup(tf.constant('<eos>')),tf.int32)
    src_dataset=src_dataset.map(lambda src:tf.string_split([src]).values)
    if src_max_len:
        src_dataset=src_dataset.map(lambda src:src[:src_max_len])
    src_dataset=src_dataset.map(lambda src:tf.cast(src_vocab_table.lookup(src),tf.int32))
    src_dataset=src_dataset.map(lambda src:tf.reverse(src,axis=[0]))
    src_dataset=src_dataset.map(lambda src:(src,tf.size(src)))
    def batching_func(x):
        return x.padded_batch(batch_size,padded_shapes=(tf.TensorShape([None]),tf.TensorShape([])),padding_values=(src_eos_id,0))
    batched_dataset=batching_func(src_dataset)
    batched_iter=batched_dataset.make_initializable_iterator()
    (src_ids,src_seq_len)=batched_iter.get_next()
    return BatchedInput(initializer=batched_iter.initializer,
        source=src_ids,
        target_input=None,
        target_output=None,
        source_sequence_length=src_seq_len,
        target_sequence_length=None)