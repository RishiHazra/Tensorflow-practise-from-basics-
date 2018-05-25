# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 09:05:31 2018
@author: Rishi
"""

import time
import math
import json
import os
import numpy as np
import pickle
from nltk.corpus import gutenberg
import tensorflow as tf
from char_lstm_model import CharLSTM
batch_size=50
seq_length=50

'''Preprocessed_model'''

def process_text(raw_text):
    import re
    raw_text=re.sub('  ',' ',raw_text)
    raw_text=re.sub ('[^A-Za-z0-9?&\.!\'\; \-\,]', '@', raw_text)
    return raw_text


def split_data(text):
    train=[]
    test=[]
    text=nltk.tokenize.sent_tokenize(text)
    split=math.floor(80*len(text)/100)
    train=text[:split]
    test=text[split:]
    return train,test


print('Pre-processing...')
start_time = time.time()
import nltk
classes=gutenberg.fileids()
classes=classes[3:8]


train_data=[]
test_data=[]
for cat in classes:
    a=[]
    b=[]
    a,b = split_data(process_text(gutenberg.raw(cat).lower()))
    train_data+=a
    test_data+=b

train_data = " ".join(train_data)
test_data = " ".join(test_data)


# dumping the train and the test splits
json_content = json.dumps({"train_data": train_data,"test_data": test_data}, indent=4)
with open('data.json','w') as outfile:
    json.dump(json_content,outfile)
    outfile.close()


# mapping every unique character to a unique index
chars = sorted(list(set(train_data)))

with open('character_set.pkl','wb') as outfile:
    pickle.dump(chars,outfile)
    outfile.close()
    
vocab_size = len(chars)
train_len = len(train_data)
char_to_value = dict((c, i) for i,c in enumerate(chars))
character_value_set = [char_to_value[c] for c in train_data]

print('Total Unique Characters:',len(chars))
print('Total Characters:' ,train_len)

## splitting data into batches
n_batches = int((len(character_value_set) - 1) / (batch_size * seq_length))
limit = n_batches * batch_size * seq_length
data_x = character_value_set[:limit]
data_y = character_value_set[1:limit + 1]
batch_x = np.split(np.reshape(data_x, [batch_size, -1]), n_batches, 1)
batch_y = np.split(np.reshape(data_y, [batch_size, -1]), n_batches, 1)


[b,c]=[np.array(batch_x).tolist(), np.array(batch_y).tolist()]
batch = json.dumps({"batch_x": b, "batch_y": c,"num_batches":n_batches}, indent=4)
with open('batch.json', 'w') as outfile:
    outfile.write(batch)
    outfile.close()


end_time = time.time()
print('Finished in %d minutes %d seconds' % ((end_time - start_time) / 60, (end_time - start_time) % 60))
def get_batch(b):
    return batch_x[b], batch_y[b]




import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default='saveChar', help='directory to store checkpointed models')
parser.add_argument('--log_dir', type=str, default='logs', help='directory to store tensorboard logs')
parser.add_argument('--lstm_size', type=int, default=128, help='size of LSTM hidden state')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers in the RNN')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
parser.add_argument('--seq_length', type=int, default=50, help='LSTM sequence length')
parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--eta', type=float, default=0.002, help='learning rate')
parser.add_argument('--decay', type=float, default=0.97, help='decay rate')
parser.add_argument('--save_every', type=int, default=1000, help='save frequency')
args = parser.parse_args()
args.vocab_size = vocab_size

with open('args.pckl','wb') as outfile1:
    pickle.dump(args,outfile1)
    outfile1.close()


## training the model

def train(args):
    model = CharLSTM(args,training=True)
    with open('batch.json', 'r') as infile:
        batch = json.load(infile)
        infile.close()
        
    batch_x=batch['batch_x']
    batch_y=batch['batch_y']
    num_batches=batch['num_batches']
    init = tf.global_variables_initializer()
    tf_saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(init)
        for e in range(args.num_epochs):
            sess.run(tf.assign(model.eta, args.eta * (args.decay ** e)))
            state = sess.run(model.initial_state)
            for b in range(num_batches):
                x, y = batch_x[b], batch_y[b]
                feed = {model.input_data: x, model.output_data: y}
                for i, (c, h) in enumerate(model.initial_state):
                    feed[c] = state[i].c
                    feed[h] = state[i].h
                _ , train_loss, state, predicted_output = sess.run([model.optimizer, model.cost, model.final_state, model.predicted_output], feed)
                accuracy = np.sum(np.equal(y, predicted_output)) / float(args.seq_length*args.batch_size)
                print("{}/{} - epoch {} , loss = {:.3f}, accuracy = {}".format(e * num_batches + b, args.num_epochs * num_batches, e, train_loss, accuracy))
                if (e * num_batches + b) % args.save_every == 0 or (e == args.num_epochs - 1 and b == num_batches- 1):
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    tf_saver.save(sess, checkpoint_path, global_step=e * num_batches + b)
                    print("Model saved to {}".format(checkpoint_path))

train(args)

args.batch_size=1
tf.reset_default_graph()
model1=CharLSTM(args,training=True)
model1.test_perplexity(test_data)


#plt.subplot(2, 3, 1)
#plt.cla()
#plt.plot(loss)
#for batch_series_idx in range(5):
#    one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
#    single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])
#
#plt.subplot(2, 3, batch_series_idx + 2)
#plt.cla()
#plt.axis([0, truncated_backprop_length, 0, 2])
#left_offset = range(truncated_backprop_length)
#plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
#plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
#plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")


#tf.reset_default_graph()
#model=CharLSTM(args,training=False)
#model.test_perplexity(test_data)
#clm.CharLSTM.test_perplexity(test_data)
#clm.