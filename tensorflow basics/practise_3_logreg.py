# -*- coding: utf-8 -*-
"""
log regression: lec 3
@author: Rishi
"""
import time
import os
os.environ['Tf_CPP_MIN_LOG_LEVEL']='2'

batch_size=128
n_epochs=30
learning_rate=0.01

import tensorflow as tf

#train_data=tf.data.Dataset.from_tensor_slices(train)
#train_data=train_data.shuffle(1000)
#train_data=train_data.batch(batch_size)
#
#test_data=tf.data.Dataset.from_tensor_slices(test)
#test_data=test_data.batch(batch_size)


X=tf.placeholder(shape=[batch_size,784],dtype=tf.float32, name='image')
Y=tf.placeholder(shape=[batch_size,10],name='label',dtype=tf.int32)

W=tf.get_variable(name='weights',dtype=tf.float32,shape=(784,10),initializer=tf.random_normal_initializer())
b=tf.get_variable(name='bias', dtype=tf.float32, shape=10, initializer= tf.random_normal_initializer())

logits= tf.add(tf.matmul(X,W),b)
entropy= tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits= logits, name='loss')
loss= tf.reduce_mean(entropy)
opt=tf.train.AdamOptimizer(learning_rate= learning_rate)

optimizer= opt.minimize(loss)

preds= tf.nn.softmax(logits)
correct_preds= tf.equal(tf.argmax(preds,1), tf.argmax(Y,1))

# tf.reduce_sum computes sum across dimensions
# x = tf.constant([1.8, 2.2], dtype=tf.float32); tf.cast(x, tf.int32)  # [1, 2], dtype=tf.int32
accuracy= tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer= tf.summary.FileWriter('./graphs/logreg',tf.get_default_graph())
with tf.Session as sess:
    start=time.time()
    sess.run(tf.variables_initializer([W,b]))
    n_batches= int(mnist.train.num_examples/batch_size)
    
    # train the model n_epochs times
    for i in range(n_epochs):
        total_loss=0
        for j in range(n_batches):
            x,y=mnist.train.next_batch(batch_size)
            _, l=sess.run([optimizer,loss],feed_dict={X:x, Y:y})
            total_loss+=l
        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
    print('Total time: {0} seconds'.format(time.time()-start))
    
    # test the model
    n_batches = int(mnist.test.num_examples/batch_size)
    total_correct_preds=0
    
    for i in range(n_batches):
        x,y= mnist.test.next_batch(batch_size)
        accuracy= sess.run([accuracy],feed_dict={X:x, Y:y})
        total_correct_preds+=accuracy

    print('Accuracy {0}'.format(total_correct_preds/mnist_test_num_examples))
    
writer.close()
        
