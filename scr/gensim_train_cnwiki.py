import clean_wiki as cw
import gensim
from time import time
import multiprocessing
import os

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname),encoding='utf8'):
                yield line.split()

if __name__ == '__main__':
    # numbers = ['test', '00', '01']
    # number = numbers[1]
    token_path = '../data_cn/'
    # token_path = '../data/news_texts_tokens.json'
    sentences = MySentences(token_path)  # a memory-friendly iterator
    # sentences = cw.open_texts(token_path)
    # begin = time()
    # model = gensim.models.Word2Vec(sentences,
    #                                size=400,
    #                                window=5,
    #                                min_count=5,
    #                                workers=multiprocessing.cpu_count())
    # model.save("../model/word2vec_normal_full.model")
    # model.wv.save_word2vec_format("../model/word2vec_normal_full.vector",
    #                               "../model/vocabulary_normal_full",
    #                               binary=False)
    #
    # end = time()
    # print("model-1 procesing time: %d seconds" % (end - begin))

    begin_2 = time()
    model_selevtive = gensim.models.Word2Vec(sentences,
                                   size=400,
                                   window=5,
                                   min_count=10000, # choose the words which minimum count is 1000
                                   workers=multiprocessing.cpu_count())
    model_selevtive.save("../model/word2vec_selective_10000.model")
    model_selevtive.wv.save_word2vec_format("../model/word2vec_selective_10000.vector",
                                  "../model/vocabulary_selective_10000",
                                  binary=False)
    end_2 = time()
    print("model-2 procesing time: %d seconds" % (end_2 - begin_2))

