# -*- coding:utf-8 -*-
'''
	Program:
		test conv using https://blog.csdn.net/tsyccnh/article/details/87357447
		only for channel and batch size is 1
	Release:
		v1.0	2019/08/27	ZhangDao
			: touch this file
'''
import numpy as np
import tensorflow as tf

batch_size = 1
stride = 1
padding = "VALID"

input_channel = 1
input_size = 5

output_channel = 1

filter_size = 3

output_size = 3

################## using Numpy ##################
print ("\n################## using Numpy ##################")
input_np = np.reshape(np.arange(input_size*input_size, dtype="float32"), newshape=[input_size, input_size])
print (f"input_np = {input_np}")
print ("input_np.shape = %s" % str(input_np.shape))

input_np_flattern = np.reshape(input_np, newshape=[input_size*input_size, 1])
print (f"input_np_flattern = {input_np_flattern}")
print ("input_np_flattern.shape = %s" % str(input_np_flattern.shape))

filter_np = np.reshape(np.arange(filter_size*filter_size, dtype="float32"), newshape=[filter_size, filter_size])
print (f"filter_np = {filter_np}")
print ("filter_np.shape = %s" % str(filter_np.shape))

filter_np_matrix = np.zeros((output_size, output_size, input_size, input_size))
for h in range(output_size):
	for w in range(output_size):
		start_h = h*stride
		start_w = w*stride
		end_h = start_h + filter_size
		end_w = start_w + filter_size
		filter_np_matrix[h, w, start_h:end_h, start_w:end_w] = filter_np

filter_np_matrix = np.reshape(filter_np_matrix, newshape=[output_size*output_size, input_size*input_size])
print (f"filter_np_matrix = {filter_np_matrix}")
print ("filter_np_matrix.shape = %s" % str(filter_np_matrix.shape))

output_np = np.dot(filter_np_matrix, input_np_flattern)
output_np = np.reshape(output_np, newshape=[output_size, output_size])
print (f"output_np = {output_np}")
print ("output_np.shape = %s" % str(output_np.shape))

output_np_flattern = np.reshape(output_np, newshape=[output_size*output_size, 1])
output_np_transpose = np.dot(filter_np_matrix.T, output_np_flattern)
output_np_transpose = np.reshape(output_np_transpose, newshape=[input_size, input_size])
print (f"output_np_transpose = {output_np_transpose}")
print ("output_np_transpose.shape = %s" % str(output_np_transpose.shape))

################## using TF ##################
print ("\n################## using TF ##################")
input_tf = tf.reshape(tf.range(batch_size*input_size*input_size*input_channel, dtype="float32"), shape=[batch_size, input_size, input_size, input_channel])
filter_tf = tf.reshape(tf.range(filter_size*filter_size*input_channel*output_channel, dtype="float32"), shape=[filter_size, filter_size, input_channel, output_channel])
output_tf = tf.nn.conv2d(input_tf, filter_tf, strides=[1, stride, stride, 1], padding=padding)
print(output_tf.get_shape())
output_shape = [batch_size, input_size, input_size, input_channel]
output_tf_transpose = tf.nn.conv2d_transpose(output_tf, filter_tf, output_shape, [1, stride, stride, 1], padding)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	print("input_tf = %s" % sess.run(input_tf))
	print("filter_tf = %s" % sess.run(filter_tf))
	print("output_tf = %s" % sess.run(output_tf))
	print("output_tf_transpose = %s" % sess.run(output_tf_transpose))
