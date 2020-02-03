import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.decomposition import PCA

from gensim.test.utils import datapath,get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = datapath('/Users/zhoup/wordEmbedding/glove/glove.6B.200d.txt')
word2vec_glove_file = get_tmpfile('glove.6B.200d.word2vec.txt')
glove2word2vec(glove_file,word2vec_glove_file)

model = KeyedVectors.load_word2vec_format(word2vec_glove_file)

print(model.most_similar('obama'))

print(model.doesnt_match(['good','bad','lovely','smart']))

from nltk.corpus import reuters

reuters.fileids()

