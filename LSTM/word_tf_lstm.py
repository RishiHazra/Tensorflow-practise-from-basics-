# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 09:05:31 2018
@author: Rishi
"""

import time
import math
import json
import tensorflow as tf
import numpy as np
import os
from collections import Counter
import pickle
from nltk.corpus import gutenberg
from word_lstm_model import WordLSTM

batch_size = 20
seq_length = 20

'''Preprocessed_model'''

#def process_text(raw_text):
#    table = str.maketrans('', '', '→îé*&`%_@~è>')
#    raw_text = raw_text.translate(table)
#    return raw_text

def process_text(raw_text):
    import re
    raw_text=re.sub('  ',' ',raw_text)
    raw_text=re.sub ('[^A-Za-z0-9?&\.!\'\; \-\,]', '', raw_text)
    return raw_text


def split_data(text):
    train = []
    test = []
    text = nltk.tokenize.sent_tokenize(text)
    split = math.floor(80 * len(text) / 100)
    train = text[:split]
    test = text[split:]
    return train, test, text

def wordToFrequency(text):
    freq_dist=Counter(text)
    freq_dist=[word for word in freq_dist.keys() if freq_dist[word]>1 or np.random.choice(2,1,p=[0.2,0.8])[0]==1 ]
    return freq_dist

def replace_withUNK(text,freq_list):
    text=[word if word in freq_list else '<UNK>' for word in text]
    " ".join(text)
    return text
    
print('Pre-processing...')
start_time = time.time()
import nltk

classes = gutenberg.fileids()
classes=classes[10:15]

train_data = []
test_data = []
whole_corpus=[]
for cat in classes:
    a = []
    b = []
    c=[]
    a, b , c= split_data(process_text(gutenberg.raw(cat).lower()))
    train_data += a
    test_data += b
    whole_corpus+=c
    
train_data = " ".join(train_data)
test_data = " ".join(test_data)
#whole_corpus= " ".join(whole_corpus)
#
#train_data=replace_withUNK(nltk.word_tokenize(train_data),wordToFrequency(nltk.word_tokenize(whole_corpus)))
#test_data=replace_withUNK(nltk.word_tokenize(test_data),wordToFrequency(nltk.word_tokenize(whole_corpus)))
#
#train_data = " ".join(train_data)
#test_data = " ".join(test_data)

# dumping the train and the test splits

json_content = json.dumps({"train_data": train_data, "test_data": test_data}, indent=4)
with open('data_word.json', 'w') as outfile:
    json.dump(json_content, outfile)
    outfile.close()

# mapping every unique word to a unique index
    
words=nltk.word_tokenize(train_data)
words=sorted(set(words))
with open('words.json','w') as outfile:
    json.dump(words,outfile)
    outfile.close()
vocab_size = len(words)
train_len = len(train_data)
word_to_value = dict((c, i) for i, c in enumerate(words))
word_value_set = [word_to_value[c] for c in words]

print('Total Unique Words:', len(words))
print('Total Words:', train_len)

## splitting data into batches

n_batches = int((len(word_value_set) - 1) / (batch_size * seq_length))
limit = n_batches * batch_size * seq_length
data_x = word_value_set[:limit]
data_y = word_value_set[1:limit + 1]
batch_x = np.split(np.reshape(data_x, [batch_size, -1]), n_batches, 1)
batch_y = np.split(np.reshape(data_y, [batch_size, -1]), n_batches, 1)

[b, c] = [np.array(batch_x).tolist(), np.array(batch_y).tolist()]
batch = json.dumps({"batch_x": b, "batch_y": c, "num_batches": n_batches}, indent=4)
with open('batch_word.json', 'w') as outfile:
    outfile.write(batch)
    outfile.close()

end_time = time.time()
print('Finished in %d minutes %d seconds' % ((end_time - start_time) / 60, (end_time - start_time) % 60))


def get_batch(b):
    return batch_x[b], batch_y[b]


import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default='saveWord', help='directory to store checkpointed models')
parser.add_argument('--log_dir', type=str, default='logs', help='directory to store tensorboard logs')
parser.add_argument('--lstm_size', type=int, default=128, help='size of LSTM hidden state')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers in the RNN')
parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
parser.add_argument('--seq_length', type=int, default=20, help='LSTM sequence length')
parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs')
parser.add_argument('--eta', type=float, default=0.002, help='learning rate')
parser.add_argument('--decay', type=float, default=0.97, help='decay rate')
parser.add_argument('--save_every', type=int, default=1000, help='save frequency')

args = parser.parse_args()
args.vocab_size = vocab_size

with open('args_word.pckl', 'wb') as outfile1:
    pickle.dump(args, outfile1)
    outfile1.close()


## training the model

def train(args):
    model = WordLSTM(args,training=True)
    with open('batch_word.json', 'r') as infile:
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

#args.batch_size=1
#tf.reset_default_graph()
#model1=WordLSTM(args,training=True)
#model1.test_perplexity(test_data)