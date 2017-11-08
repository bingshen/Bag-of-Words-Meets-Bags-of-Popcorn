import numpy as np
import tensorflow as tf

a=tf.Variable(10,dtype=tf.int32)
b=tf.Variable(20,dtype=tf.int32)
c=tf.cond(a<b,lambda: tf.constant([1,0]),lambda: tf.constant([0,1]))
print(c)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(c))