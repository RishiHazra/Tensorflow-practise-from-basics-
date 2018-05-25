# -*- coding: utf-8 -*-
"""
Assignment 1
@author: Rishi
"""
#suppress the warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

###############################################################################
# 1a: Create two random 0-d tensors x and y of any distribution.
# Create a TensorFlow object that returns x + y if x > y, and x - y otherwise.
###############################################################################

x= tf.random_normal([])
y=tf.random_normal([])

out=tf.cond(tf.greater(x,y),lambda: x+y, lambda: x-y)
sess= tf.Session()
print(sess.run([x,y,out]))


###############################################################################
# 1b: Create two 0-d tensors x and y randomly selected from the range [-1, 1).
# Return x + y if x < y, x - y if x > y, 0 otherwise.
###############################################################################

x=tf.random_uniform([],minval=-1,maxval=1, dtype=tf.float32)
y=tf.random_uniform([],minval=-1, maxval=1, dtype=tf.float32)

out=tf.case({tf.less(x, y):lambda: tf.add(x,y), tf.greater(x, y): 
    lambda:tf.subtract(x,y)}, default=lambda:tf.constant(0.0) ,exclusive=True)

sess=tf.Session()
print(sess.run([x,y,out]))

###############################################################################
# 1c: Create the tensor x of the value [[0, -2, -1], [0, 1, 2]] 
# and y as a tensor of zeros with the same shape as x.
# Return a boolean tensor that yields Trues if x equals y element-wise.
###############################################################################

x=tf.constant( [[0, -2, -1], [0, 1, 2]])
y=tf.zeros_like(x)
out=tf.equal(x,y)

###############################################################################
# 1d: Create the tensor x of value 
# [29.05088806,  27.61298943,  31.19073486,  29.35532951,
#  30.97266006,  26.67541885,  38.08450317,  20.74983215,
#  34.94445419,  34.45999146,  29.06485367,  36.01657104,
#  27.88236427,  20.56035233,  30.20379066,  29.51215172,
#  33.71149445,  28.59134293,  36.05556488,  28.66994858].
# Get the indices of elements in x whose values are greater than 30.
###############################################################################

x=tf.constant([29.05088806,  27.61298943,  31.19073486,  29.35532951,
               30.97266006,  26.67541885,  38.08450317,  20.74983215,
               34.94445419,  34.45999146,  29.06485367,  36.01657104,
               27.88236427,  20.56035233,  30.20379066,  29.51215172,
               33.71149445,  28.59134293,  36.05556488,  28.66994858])

indices=tf.where(x>30)  
out=tf.gather(x,indices)
print(sess.run(out))

###############################################################################
# 1e: Create a diagonal 2-d tensor of size 6 x 6 with the diagonal values of 1,
# 2, ..., 6
###############################################################################

values=tf.range(7)
out=tf.diag(values)

###############################################################################
# 1f: Create a random 2-d tensor of size 10 x 10 from any distribution.
# Calculate its determinant.
###############################################################################

x=tf.random_normal([10,10])    
out=tf.matrix_determinant(x)
sess.run([x,out])

###############################################################################
# 1g: Create tensor x with value [5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9].
# Return the unique elements in x

###############################################################################

x=tf.constant([5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9])
out, idx= tf.unique(x)

###############################################################################
# 1h: Create two tensors x and y of shape 300 from any normal distribution,
# as long as they are from the same distribution.
# Use tf.cond() to return:
# - The mean squared error of (x - y) if the average of all elements in (x - y)
#   is negative, or
# - The sum of absolute value of all elements in the tensor (x - y) otherwise.
###############################################################################

x=tf.random_normal([300])
y=tf.random_normal([300])
avg=tf.reduce_mean(x-y)
out=tf.cond(avg<0 ,lambda:tf.reduce_mean(tf.square(x-y))
,lambda: tf.reduce_sum(tf.abs(x-y)))