# -*- coding: utf-8 -*-
"""
Created on Tue May  8 18:06:31 2018

@author: Rishi
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
x=tf.constant(3)
y=tf.constant(5)
a=tf.add(x,y)

writer= tf.summary.FileWriter('./graphs', tf.get_default_graph())
with tf.Session() as sess:
    writer= tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(a))
    
writer.close()
'''
$ python3 practise_1.py
$ tensorboard --logdir= "./graphs" --port 6006
$ http://localhost:6006/
'''

z=tf.zeros([2,3],dtype=tf.int32,name='z')
m=tf.zeros_like(z)
with tf.Session() as sess:
    print(sess.run(z),"\n\n",sess.run(m))
    print(sess.run(tf.range(1,10,2)))
    
# tensor objects are not iterable
    
'''
tf.set_random_seed(seed)
'''

    
#x=2
#y=3
#opt1= tf.add(x,y)
#opt2= tf.multiply(x,y)  
#opt3= tf.pow(opt1,opt2)
#
#with tf.Session() as sess:
#    print(sess.run(opt3))
#    
#g=tf.Graph()
#with g.as_default():
#    x=tf.add(3,5)
#sess= tf.Session(graph=g)
#with tf.Session() as sess:
#    sess.run(x)