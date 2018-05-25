# -*- coding: utf-8 -*-
"""
Created on Wed May  9 13:56:10 2018

@author: Rishi
"""

import numpy as np
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf


def read_birth_lifes():
    text=open('birth_life_2010.txt','r').readlines()[1:]
    data= [line[:-1].split('\t') for line in text]
    birth = [line[1] for line in data]
    lifes= [line[2] for line in data]
    data= np.asarray(list(zip(birth,lifes)), dtype=np.float32)
    n_samples= len(data)
    return data, n_samples


data, n_samples= read_birth_lifes()


X= tf.placeholder(dtype=tf.float32)
Y= tf.placeholder(dtype=tf.float32)

W= tf.get_variable('weight',initializer=tf.random_uniform(shape=[1]))
b=tf.get_variable('bias',dtype=None, initializer= tf.random_uniform(shape=[1]))

Y_pred=tf.add(tf.multiply(W,X),b)

loss=tf.square(Y-Y_pred, name='loss')

opt=tf.train.GradientDescentOptimizer(learning_rate=0.001)
optimizer= opt.minimize(loss)

start=time.time()


with tf.Session() as sess:
    writer= tf.summary.FileWriter('./graphs_lin_reg', sess.graph)
    sess.run(tf.variables_initializer([W,b]))
    for i in range (100):
        total_loss=0
        for x,y in data:
            _, loss_ = sess.run([optimizer, loss], feed_dict={X:x, Y:y})
            total_loss += loss_
        print('Epoch {0}: {1}'.format(i, total_loss/n_samples))
    writer.close()
    w_out, b_out = sess.run([W, b])
    
    
print('Took: %f seconds' %(time.time() - start))

import matplotlib.pyplot as plt
plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data')
plt.legend()
plt.show()
