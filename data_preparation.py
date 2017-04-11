import tensorflow as tf
import numpy as np
import gensim
import pickle
import os
import time
import datetime
import data_helpers
from tensorflow.contrib import learn

tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the positive data.")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# # Data Preparatopn
# # ==================================================

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

# Load pretrained model
print("Loading model ...")
model = gensim.models.Word2Vec.load_word2vec_format('~/GoogleNews-vectors-negative300.bin', binary=True)  
print("Loading model finished")

# Build embedding matrix
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_dict = dict()
for sent in x_text:
	words = sent.split(" ")
	for word in words:
		if word in model.vocab:
			vocab_dict[word] = model[word];
		else:
			vocab_dict[word] = np.zeros(300)

pickle.dump(vocab_dict, open("embedding_dict.p", "wb"))

