"""
generate sentence
"""

import argparse
import pickle
from char_lstm_model import CharLSTM
import tensorflow as tf
tf.reset_default_graph()

def process_text(raw_text):
    import re
    raw_text=re.sub('\.+','\.',raw_text)
    raw_text=re.sub('  ',' ',raw_text)
    raw_text=re.sub ('[^A-Za-z0-9?&\.!\'\; \-\,]', '', raw_text)
#    raw_text=re.sub ('[^A-Za-z0-9?&\.!\'; -\,]', '@', raw_text)
    return raw_text


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default='./saveChar', help='Directory to save model checkpoints')
parser.add_argument('--start', type=str, default='the ', help='Start of the generation')
parser.add_argument('--predict', type=int, default=100, help='No of predictions')
args = parser.parse_args()

with open('args.pckl','rb') as infile:
    model_args=pickle.load(infile)
    infile.close()

   
with open('character_set.pkl','rb') as infile:
    characters=pickle.load(infile)
    infile.close()
    
model = CharLSTM(model_args,training=False)
sentence = model.generate(characters, process_text(args.start), args.predict)
print(sentence)
