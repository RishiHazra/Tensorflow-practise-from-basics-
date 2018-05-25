# -*- coding: utf-8 -*-
"""
Created on Wed May  9 09:56:04 2018

@author:Rishi
"""

import tensorflow as tf

W= tf.Variable(tf.zeros([784,10]),'big_matrix')
with tf.Session() as sess:
    sess.run(W.initializer)
    print(W.eval())

# tf.fill([2,3],8)





# get_variable help us in variable sharing by making (reuse=True) 
# which is not the case with tf.varible
s=tf.get_variable('scalar',initializer= tf.constant(2))
m=tf.get_variable('matrix', initializer= tf.constant([[1,2],[3,4]]))
w= tf.get_variable('big_matix', shape=(784,10), initializer=tf.zeros_initializer())
j= tf.get_variable('big',shape=(784,10), initializer= tf.truncated_normal_initializer())

with tf.Session() as sess:
#    print(sess.run(tf.global_variables_initializer()))
    sess.run(tf.variables_initializer([s,m,j]))
    print(j.eval())
    
    
    
'''
assign
'''
my_var=tf.get_variable('my_var', initializer=tf.constant(10))

with tf.Session() as sess:
    sess.run(my_var.initializer)
    print(my_var.eval())
    print(my_var.assign_add(10).eval())
    print(my_var.assign_sub(5).eval())