#!/usr/bin/python
# -*- coding:utf-8 -*-


'''
import tensorflow as tf

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
    a = tf.constant(1)
    b = tf.constant(3)
    c = a + b
    print('结果是：%d\n 值为：%d' % (sess.run(c), sess.run(c)))



saver = tf.train.Saver()
model_path_pb = "."



with tf.Session() as sess:
    sess.run(init)
saver_path = saver.save(sess,model_path_pb)


#保存成pb文件

constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['c'])

with tf.gfile.FastGFile(model_path_pb +'model1.pb', mode='wb') as f:
    f.write(constant_graph.SerializeToString())



'''


import tensorflow as tf
 
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2
 
saver = tf.train.Saver()
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, "./model.ckpt")

