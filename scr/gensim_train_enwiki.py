import clean_enwiki as cw
import gensim
from time import time
import multiprocessing
import os
import argparse

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder 'model' created !  ---")
    return

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname),encoding='utf8'):
                yield line.split()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--model_version', type=str, nargs='?',
                        default='enfull',
                        help='the version of the model:full, or selective_number')
    parser.add_argument('-mn', '--minimum_count', type=int, nargs='?',
                        default='5',
                        help='the mumimum frequency of each word, default = 5')

    args = parser.parse_args()
    model_version = args.model_version
    minimum_count = args.minimum_count

    token_path = '../data/'
    sentences = MySentences(token_path)  # a memory-friendly iterator
    begin = time()
    model = gensim.models.Word2Vec(sentences,
                                   size=400,
                                   window=5,
                                   min_count=minimum_count,
                                   workers=multiprocessing.cpu_count()
                                   )
    model.save("../model/word2vec_{}.model".format(model_version))
    model.wv.save_word2vec_format("../model/word2vec_{}.vector".format(model_version),
                                  "../model/word2vec_vocabulary_{}".format(model_version),
                                  binary=False)
    
    end = time()
    print("model-{} procesing time: {} seconds".format(model_version,end - begin))

    # begin_2 = time()
    # model_selevtive = gensim.models.Word2Vec(sentences,
    #                                size=400,
    #                                window=5,
    #                                min_count=10000, # choose the words which minimum count is 10000
    #                                workers=multiprocessing.cpu_count())    
    # mkdir('../model')
    # model_selevtive.save("../model/word2vec_selective_10000.model")
    # model_selevtive.wv.save_word2vec_format("../model/word2vec_selective_10000.vector",
    #                               "../model/vocabulary_selective_10000",
    #                               binary=False)
    # end_2 = time()
    # print("model-2 procesing time: %d seconds" % (end_2 - begin_2))

