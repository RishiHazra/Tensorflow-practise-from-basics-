# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 23:24:01 2018

@author: Rishi
"""

"""
generate sentence
"""
import argparse
import json
import pickle
import nltk
from word_lstm_model import WordLSTM


def process_text(raw_text):
    import re
    raw_text=re.sub('  ',' ',raw_text)
    raw_text=re.sub ('[^A-Za-z0-9?&\.!\'\; \-\,]', '', raw_text)
    raw_text
    words=nltk.word_tokenize(raw_text)
    return words


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default='./saveWord', help='Directory to save model checkpoints')
parser.add_argument('--start', type=str, default='that', help='Start of the generation')
parser.add_argument('--predict', type=int, default=10, help='No of predictions')
args = parser.parse_args()

with open('args_word.pckl','rb') as infile:
    model_args=pickle.load(infile)
    infile.close()

with open('words.json','r') as infile:
    words=json.load(infile)
    infile.close()
    
model = WordLSTM(model_args,training=False)
print(" ".join(model.generate(words,process_text(args.start), args.predict)))



