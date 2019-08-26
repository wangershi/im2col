# -*- coding:utf-8 -*-
'''
	Program:
		test conv
	Release:
		v1.0	2019/08/26	ZhangDao
			: touch this file
'''
import tensorflow as tf
import numpy as np

class Batch2ConvMatrix:
	def __init__(self, stride, kernel_h, kernel_w):
		self.stride = stride
		self.kernel_h = kernel_h
		self.kernel_w = kernel_w

		self.x = None
		self.conv_size = None

	def __call__(self, x):
		self.x = x
		x_batch, x_height, x_width, x_channels = np.shape(self.x)
		
		conv_height = int((x_height - self.kernel_h) / self.stride) + 1
		conv_width = int((x_width - self.kernel_w) / self.stride) + 1

		scan = np.zeros((x_batch, conv_height, conv_width,
			x_channels, self.kernel_h, self.kernel_w))
		for n in range(x_batch):
			for h in range(conv_height):
				for w in range(conv_width):
					for c in range(x_channels):
						start_h = h * self.stride
						start_w = w * self.stride
						end_h = start_h + self.kernel_h
						end_w = start_w + self.kernel_w

						scan[n, h, w, c] = \
							x[n, start_h:end_h, start_w:end_w, c]

		conv_matrix = scan.reshape(x_batch * conv_height * conv_width, -1)
		self.conv_size = [x_batch, conv_height, conv_width, x_channels]
		return conv_matrix

batch_size = 1
stride = 2
padding = "VALID"

input_channel = 2
input_size = 5

output_channel = 2

filter_size = 3

output_size = 2

################## using Numpy ##################
print ("\n################## using Numpy ##################")
input_np = np.reshape(np.arange(batch_size*input_size*input_size*input_channel, dtype="float32"), newshape=[batch_size, input_size, input_size, input_channel])
print (f"input_np = {input_np}")
print ("input_np.shape = %s" % str(input_np.shape))
x2m = Batch2ConvMatrix(stride, filter_size, filter_size)(input_np)
print (f"x2m = {x2m}")
print ("x2m.shape = %s" % str(x2m.shape))

filter = np.reshape(np.arange(filter_size*filter_size*input_channel*output_channel, dtype="float32"), newshape=[filter_size, filter_size, input_channel, output_channel])
print (f"filter = {filter}")
print ("filter.shape = %s" % str(filter.shape))

filter2m = np.zeros((output_channel, filter_size*filter_size*input_channel))
for o in range(output_channel):
	for i in range(input_channel):
		for h in range(filter_size):
			for w in range(filter_size):
				filter2m[o][i*filter_size*filter_size+h*filter_size+w] = filter[h][w][i][o]
print (f"filter2m = {filter2m}")
print ("filter2m.shape = %s" % str(filter2m.shape))

output = np.dot(x2m, filter2m.T)
output = output.reshape([batch_size, output_size, output_size, output_channel])
print (f"output = {output}")
print ("output.shape = %s" % str(output.shape))

################## using TF ##################
print ("\n################## using TF ##################")
input_tf = tf.reshape(tf.range(batch_size*input_size*input_size*input_channel, dtype="float32"), shape=[batch_size, input_size, input_size, input_channel])
filter_tf = tf.reshape(tf.range(filter_size*filter_size*input_channel*output_channel, dtype="float32"), shape=[filter_size, filter_size, input_channel, output_channel])
output_tf = tf.nn.conv2d(input_tf, filter_tf, strides=[1, stride, stride, 1], padding=padding)
print(output_tf.get_shape())

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	print("input_tf = %s" % sess.run(input_tf))
	print("filter_tf = %s" % sess.run(filter_tf))
	print("output_tf = %s" % sess.run(output_tf))
