import tensorflow as tf

def test_func(a,b,c):
    ta=tf.cast(tf.constant([1,2]),dtype=tf.int32)
    tb=tf.cast(tf.constant([3,4]),dtype=tf.int32)
    result=tf.get_variable(name='result',shape=(2),dtype=tf.int32,initializer=tf.constant_initializer([0,0]))
    result=(ta+tb)*c
    print(c)
    return result

if __name__ == '__main__':
    result=test_func([1,2],[3,4],2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(result))
        print(sess.run(result))