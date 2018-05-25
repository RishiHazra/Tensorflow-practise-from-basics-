# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 22:51:05 2018

@author: Rishi
"""
import os
import numpy as np
import matplotlib.image as mpimg
folders = ['boxing', 'handwaving', 'running_train']

# converting rgb to gray
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

images={}
for fl in folders:
    dirname = fl+'/'
    print(fl)
    images[fl]=[]
    for filename in os.listdir(fl):
        path=os.path.join(fl,filename)
        count=1
        for f2 in os.listdir(path):            
            image = mpimg.imread(os.path.join(path, f2))
            image= rgb2gray(image) 
            if image is not None and count<=100:
                images[fl].append(image)
            else:
                break
            count+=1
   
    
# train and test splits
train={}; test={}

train['boxing']= images['boxing'][0:3200]
train['handwaving']= images['handwaving'][0:3200]
train['running_train']= images['running_train'][0:3200]
test['boxing']= images['boxing'][-800:]
test['handwaving']= images['handwaving'][-800:]
test['running_train']= images['running_train'][-800:]

Tr=train['boxing']+train['handwaving']+train['running_train']
Tr_Y=['boxing']*3200 + ['handwaving']*3200 + ['running_train']*3200

Te=test['boxing']+test['handwaving']+test['running_train']
Te_Y=['boxing']*800 + ['handwaving']*800 + ['running_train']*800

import pandas as pd
Tr_Y=pd.get_dummies(Tr_Y).values
Te_Y=pd.get_dummies(Te_Y).values

import tensorflow as tf
from tensorflow.contrib import rnn

#define constants
#unrolled through 120 time steps
time_steps=120
#hidden LSTM units
num_units=128
#rows of 160 pixels
n_input=160
#learning rate for adam
learning_rate=0.008
#mnist is meant to be classified in 10 classes(0-9).
n_classes=3
#size of batch
batch_size=3200


#weights and biases of appropriate shape to accomplish above task
out_weights=tf.Variable(tf.random_normal([num_units,n_classes]))
out_bias=tf.Variable(tf.random_normal([n_classes]))

#defining placeholders
#input image placeholder
x=tf.placeholder("float",[None,time_steps,n_input])
#input label placeholder
y=tf.placeholder("float",[None,n_classes])

#processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
input=tf.unstack(x ,time_steps,1)

#defining the network
lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
outputs,_=rnn.static_rnn(lstm_layer,input,dtype="float32")

#As we are concerned only with input of last time step, we will generate our prediction out of it.
#converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
prediction=tf.matmul(outputs[-1],out_weights)+out_bias


#loss_function
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
#optimization
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#model evaluation
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


def next_batch(num, data, labels):
    '''
    Return a total of 'num' random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i,:] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


#initialize variables
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    iter=1
    while iter<150:
        print("iter {}...".format(iter))
        batch_x, batch_y=next_batch(batch_size, Tr, Tr_Y)       
        batch_x=batch_x.reshape((batch_size,time_steps,n_input))

        sess.run(opt, feed_dict={x: batch_x, y: batch_y})

        if iter %10==0:
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            los=sess.run(loss,feed_dict={x:batch_x,y:batch_y})
            print("For iter ",iter)
            print("Accuracy ",acc)
            print("Loss ",los)
            print("__________________")
            # test accuracy
            #calculating test accuracy
            test_data1,test_label1 = next_batch(128, Te, Te_Y)
            print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data1, y: test_label1}))
            print("__________________")
        iter=iter+1

# loss, training accuracy and testing accuracy every 10 epochs       
Loss=[2.442,1.3459, 1.236, 1.109, 1.126, 1.099, 1.100, 1.099, 1.098,1.098, 1.098, 1.098, 1.098, 1.098]  
training_acc=[0.3328, 0.3375, 0.33625, 0.3231, 0.3372, 0.33, 0.329, 0.330, 0.3356,0.338, 0.3375, 0.343, 0.325, 0.336 ]      
test_accuracy=[0.2421, 0.28125, 0.3359, 0.3281,0.2968, 0.32, 0.320, 0.3828, 0.320, 0.289, 0.3515, 0.3671, 0.3671, 0.3672 ]

import matplotlib.pyplot as plt
f,(ax1,ax2)=plt.subplots(2, sharex=True)
ax1.plot(Loss,'r')
#plt.plot(training_acc, 'b')
ax2.plot(test_accuracy, 'g')
plt.xlabel('per 10 epochs')
ax1.set_ylabel('Training Loss')
ax2.set_ylabel('Test accuracy')
