################################################################################
#                              word LSTM MODEL                                 #
''' author: Rishi'''
''' NLU Assignment 2'''
################################################################################

import tensorflow as tf
import nltk
import json
from tensorflow.contrib import rnn, legacy_seq2seq
import numpy as np
tf.reset_default_graph()
import matplotlib.pyplot as plt


class WordLSTM:
    def __init__(self, args, training=True):
        self.args = args
        if not training:
            args.batch_size = 1
            args.seq_length = 1

        def lstm_cell(lstm_size):
            return tf.contrib.rnn.BasicLSTMCell(lstm_size)

        cells = []
        for i in range(args.num_layers):
            cells.append(lstm_cell(args.lstm_size))

        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.output_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('lstm'):
            softmax_w = tf.get_variable("softmax_w", [args.lstm_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])

        embedding = tf.get_variable("embedding", [args.vocab_size, args.lstm_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        inputs = tf.split(inputs, args.seq_length, 1)  # splits the input into subtensor sequences dimension 1
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell)
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.lstm_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        self.predicted_output = tf.reshape(tf.argmax(self.probs, 1), [args.batch_size, args.seq_length])

        ## loss definition
        loss = legacy_seq2seq.sequence_loss_by_example([self.logits],
            [tf.reshape(self.output_data, [-1])],
            [tf.ones([args.batch_size * args.seq_length])])
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.eta = tf.Variable(0.0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.eta).minimize(self.cost)
        
        
#    def generate(self, character_set, start, num_predictions):
#        args = self.args
#        init = tf.global_variables_initializer()
#        char_to_value = dict((c, i) for i, c in enumerate(character_set))
#        value_to_char = dict((i, c) for i, c in enumerate(character_set))
#        sentence = start
#        with tf.Session() as sess:
#            sess.run(init)
#            init = tf.initialize_all_variables()
#            sess.run(init)
#            tf_saver = tf.train.Saver(tf.global_variables())
#            checkpoint = tf.train.get_checkpoint_state(args.save_dir)
#            if checkpoint and checkpoint.model_checkpoint_path:
#                tf_saver.restore(sess, checkpoint.model_checkpoint_path)
#            state = sess.run(self.cell.zero_state(1, tf.float32))
#            for c in start[:-1]:
#                x = np.reshape(char_to_value[c], [1, 1])
#                feed = {self.input_data: x, self.initial_state: state}
#                state = sess.run(self.final_state, feed)
#
#            c = start[-1]
#            for i in range(num_predictions):
#                x = np.reshape(char_to_value[c], [1, 1])
#                feed = {self.input_data: x, self.initial_state: state}
#                prob, state = sess.run([self.probs, self.final_state], feed)
#                val = int(np.argmax(prob[0]))
#               
#                
#                import random
#                rand=random.uniform(0,1)
#                while np.amax(prob[0]) < rand:
#                   # print(np.amax(prob[0]))
#                   # print(value_to_char[val])
#                    c = value_to_char[val]
#                    x = np.reshape(char_to_value[c], [1, 1])
#                    feed = {self.input_data: x, self.initial_state: state}
#                    prob, state = sess.run([self.probs, self.final_state], feed)
#                    val = int(np.argmax(prob[0]))
#                    rand=0.01*random.uniform(0,1)                    
#                c = value_to_char[val]
#                sentence += [c]
#            return sentence
        
        
    def generate(self, character_set, start, num_predictions):
        args = self.args
        init = tf.global_variables_initializer()
        char_to_value = dict((c, i) for i, c in enumerate(character_set))
        value_to_char = dict((i, c) for i, c in enumerate(character_set))
        sentence = start
        with tf.Session() as sess:
            sess.run(init)
            init = tf.initialize_all_variables()
            sess.run(init)
            tf_saver = tf.train.Saver(tf.global_variables())
            checkpoint = tf.train.get_checkpoint_state(args.save_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                tf_saver.restore(sess, checkpoint.model_checkpoint_path)
            state = sess.run(self.cell.zero_state(1, tf.float32))
            for c in start[:-1]:
                x = np.reshape(char_to_value[c], [1, 1])
                feed = {self.input_data: x, self.initial_state: state}
                state = sess.run(self.final_state, feed)

            c = start[-1]
            for i in range(num_predictions):
                x = np.reshape(char_to_value[c], [1, 1])
                feed = {self.input_data: x, self.initial_state: state}
                prob, state = sess.run([self.probs, self.final_state], feed)
                if c == ' ':
                    val = int(np.searchsorted(np.cumsum(prob[0]), np.random.rand(1)))
                else:
                    val = int(np.argmax(prob[0]))
                c = value_to_char[val]
                sentence += [c]
            return sentence
        
        
    
    def test_perplexity(self, test_data):
        args=self.args
        with open('words.json','r') as infile:
            chars=json.load(infile)
            infile.close()
        init =tf.global_variables_initializer()
        test_data=nltk.word_tokenize(test_data)
        char_to_value = dict((c, i) for i, c in enumerate(chars))
        test_value_set=np.array([char_to_value[c] for c in test_data])
        n_batches=int((len(test_data)-1 )/ args.seq_length)
        limit = n_batches * args.seq_length
        data_x = np.reshape(test_value_set[:limit], [n_batches, args.seq_length])
        data_y = np.reshape(test_value_set[1:limit + 1], [n_batches, args.seq_length])
        with tf.Session() as sess:
            sess.run(init)
            tf_saver = tf.train.Saver(tf.global_variables())
            checkpoint = tf.train.get_checkpoint_state(args.save_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                tf_saver.restore(sess, checkpoint.model_checkpoint_path)
            state = sess.run(self.cell.zero_state(1, tf.float32))
            ppl = 0
            for i in range(n_batches):
                seq = np.reshape(data_x[i, :], [args.batch_size, args.seq_length])
                feed = {self.input_data: seq, self.initial_state: state}
                prob, state = sess.run([self.probs, self.final_state], feed)
                prob = np.log(prob[np.arange(len(prob)), data_y[i, :]])
                ppl += np.sum(prob)
            ppl /= args.seq_length * n_batches
            ppl = np.exp(-ppl)
            return ppl


    
#        plt.subplot(2, 3, 1)
#        plt.cla()
#        plt.plot(loss)
#        for batch_series_idx in range(5):
#        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
#        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in                    one_hot_output_series])
#        
#        plt.subplot(2, 3, batch_series_idx + 2)
#        plt.cla()
#        plt.axis([0, truncated_backprop_length, 0, 2])
#        left_offset = range(truncated_backprop_length)
#        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
#        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
#        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")
