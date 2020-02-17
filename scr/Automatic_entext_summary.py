#!/usr/bin/env python
# coding: utf-8


import re
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec
import nltk
import evaluation
import os


def load_model(fname: str):
    return Word2Vec.load(fname)


class AutomaticTextSummarizer:
    def __init__(self, word_vec_model, a=1e-4, C_compute_model='cs', n_neighbors=5, summarize_sentences_size=0.3):
        """
        Parameters
        ----------
        word_vec_model: Word2Vec
            Word2Vec model
        a: float
            sentence embedding weight
        C_compute_model: str
            C_i calculation model
            'cs'
                Cosine similarity
            'cc'
                Correlation coefficient
        n_neighbors: int
            KNN smooth window size
        summarize_sentences_size: float
            summarize sentences size, should be between 0.0 and 1.0
        """
        self.word_vec_model = word_vec_model
        self.a = a
        self.C_compute_model = C_compute_model
        self.n_neighbors = n_neighbors
        self.summarize_sentences_size = summarize_sentences_size

    def summarize(self, title: str, content: str):
        """
        Parameters
        ----------
        content: str
            news content
        title: str
            news title
        Returns
        -------
        summarize: dict
            summarization: str
            C: dict
            Vs: dict
            Vt: list
            Vc: list
        """
        sentences = self.cut_sentences(content)

        all_sentence_embeddings = self.compute_sentence_embeddings([title, content] + [str(a) for a in sentences])
        Vt = all_sentence_embeddings[0]  # the vector of the title
        Vc = all_sentence_embeddings[1]  # the vector of the content as a whole
        Vs = all_sentence_embeddings[2:]  # the vector of the content, each sentence as a vector

        C = self.compute_C(Vs, Vt, Vc)

        summarize_sentences_size = int(self.summarize_sentences_size * len(sentences))
        summarize_sentences = [(sentence, index) for sentence, index, _ in
                               sorted(zip(sentences, range(len(sentences)), C), key=lambda e: e[2], reverse=True)[
                               :summarize_sentences_size]]
        summarization = ''.join([sentence for sentence, _ in sorted(summarize_sentences, key=lambda e: e[1])])

        return {
            'summarization': summarization,
            'C': dict(zip(C, sentences)),
            'Vs': dict(zip(sentences, Vs)),
            'Vc': Vc,
            'Vt': Vt
        }

    def compute_sentence_embeddings(self, sentences):
        model = self.word_vec_model
        a = self.a
        sentence_embeddings = np.array([self.compute_sentence_embedding(sentence, model, a) for sentence in sentences])
        return self.remove_pc(sentence_embeddings)

    def compute_pc(self, X, npc=1):
        """
        from https://github.com/PrincetonML/SIF/blob/master/src/SIF_embedding.py
        Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
        :param X: X[i,:] is a data point
        :param npc: number of principal components to remove
        :return: component_[i,:] is the i-th pc
        """
        svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
        svd.fit(X)
        return svd.components_

    def remove_pc(self, X, npc=1):
        """
        from https://github.com/PrincetonML/SIF/blob/master/src/SIF_embedding.py
        Remove the projection on the principal components
        :param X: X[i,:] is a data point
        :param npc: number of principal components to remove
        :return: XX[i, :] is the data point after removing its projection
        """
        pc = self.compute_pc(X, npc)
        if npc == 1:
            XX = X - X.dot(pc.transpose()) * pc
        else:
            XX = X - X.dot(pc.transpose()).dot(pc)
        return XX

    def compute_C(self, Vs, Vt, Vc):
        if self.C_compute_model == 'cs':
            C_compute_model = self.cosine_similarity
        elif self.C_compute_model == 'cc':
            C_compute_model = self.correlation_coefficient
        else:
            C_compute_model = self.cosine_similarity
        return self.KNN_smooth([C_compute_model(Vsi, Vt, Vc) for Vsi in Vs])

    def KNN_smooth(self, C):
        n_neighbors = self.n_neighbors
        smooth_C = []
        C_len = len(C)
        for i in range(C_len):
            begin_index = i - n_neighbors
            if begin_index < 0: begin_index = 0

            end_index = i + n_neighbors + 1
            if end_index > C_len: end_index = C_len

            smooth_C.append(sum(C[begin_index:end_index]) / (end_index - begin_index))

        return smooth_C

    def set_word_vec_model(self, word_vec_model):
        self.word_vec_model = word_vec_model

    def set_a(self, a):
        self.a = a

    def set_C_compute_model(self, C_compute_model):
        self.C_compute_model = C_compute_model

    def set_n_neighbors(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def set_summarize_sentences_size(self, summarize_sentences_size):
        self.summarize_sentences_size = summarize_sentences_size

    @staticmethod
    def cosine_similarity(Vsi, Vt, Vc):
        """
        余弦相似度
        """
        si_t_cs = abs(np.sum(Vsi * Vt) / (np.sqrt(np.sum(Vsi ** 2)) * np.sqrt(np.sum(Vt ** 2))))
        si_c_cs = abs(np.sum(Vsi * Vc) / (np.sqrt(np.sum(Vsi ** 2)) * np.sqrt(np.sum(Vc ** 2))))

        return (si_t_cs + si_c_cs) / 2

    @staticmethod
    def correlation_coefficient(Vsi, Vt, Vc):
        """
        相关系数
        """
        pass

    @staticmethod
    def compute_sentence_embedding(sentence, model, a):
        """
        如果词向量不存在应该如何处理？（目前的处理是忽略该词向量）(out-of-word)
        """
        words = AutomaticTextSummarizer.cut_words(sentence)
        # 词向量加权求和
        word_embeddings = np.array(
            [a / (a + (model.wv.vocab[word].count / model.corpus_total_words)) * model.wv[word] for word in words if
             word in model.wv])
        return np.sum(word_embeddings, axis=0) / word_embeddings.shape[0]

    @staticmethod
    def compute_first_principal_component(sentence_embeddings):
        pca = PCA(n_components=2)
        pca.fit(sentence_embeddings)
        return pca.components_

    @staticmethod
    def cut_words(content: str):
        return [word for word in list(AutomaticTextSummarizer.clean_data(content)) if word != ' ']

    @staticmethod
    def cut_sentences(content: str):
        '''return a list of sentences'''
        # content = AutomaticTextSummarizer.clean_data(content)
        return nltk.sent_tokenize(content, language='english')

    @staticmethod
    def clean_data(content: str):
        chinese_punctuation = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠\［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'
        english_punctuation = ',.?;:\'"`~!\[\]\-'
        special_char = r'<>/\\|\[\]{}@#\$%\^&\*\(\)-\+=_\n'
        return re.sub(
            '(?P<punctuation>[{}]|[{}])|(?P<special_char>[{}])'.format(chinese_punctuation, english_punctuation,
                                                                       special_char), ' ', content)


if __name__ == '__main__':
#     title = """ President Xi warns virus 'directly affects' economic and social stability of China"""
#     content = """ The Chinese state news agency Xinhau reports that China’s President Xi has made an important speech to the Standing Committee of the Political Bureau of the Communist Party of China Central Committee on Monday to address the coronavirus outbreak.
#
# The outcome of the epidemic prevention and control directly affects people’s lives and health, the overall economic and social stability and the country’s opening-up, Xinhau says.
#
# Xinhau says Xi demanded “resolute opposition against bureaucratism and the practice of formalities for formalities’ sake in the prevention work”.
#
# Those who disobey the unified command or shirk off responsibilities will be punished, Xi said. The report said that the party and government leaders supervising them would also be held accountable in severe cases."""
    evaluation1 = {'precision':[],'recall':[],'fmeasure':[]}
    evaluation2 = {'precision': [], 'recall': [], 'fmeasure': []}
    evaluation3 = {'precision': [], 'recall': [], 'fmeasure': []}
    topics = ['business', 'entertainment', 'politics', 'sport', 'tech']
    topic = topics[3]
    article_root = '../evaluation_data/BBC_News_Summary/News_Articles/{}'.format(topic)
    summary_root = '../evaluation_data/BBC_News_Summary/Summaries/{}'.format(topic)
    path = '../model/'
    model = load_model(path + 'word2vec_enfull.model')
    print('model', model)
    # vector = load_model(path + 'word2vec_test.vector')
    automatic_text_summarizer = AutomaticTextSummarizer(model)
    automatic_text_summarizer.set_n_neighbors(2)
    automatic_text_summarizer.set_summarize_sentences_size(0.4)

    article_lists = os.listdir(article_root)
    for s in article_lists:

        article_ = open(article_root + '/' + s, "r")
        article = article_.read().replace('\n', '.').split('.')
        title, content = evaluation.extract_title_and_content(article)
        article_.close()

        summary_ = open(summary_root + '/' + s, "r")
        summary = summary_.read().split('.')
        summary = ". ".join(summary)
        # print('summary_origin-------------\n', summary)
        summary_.close()

        result = automatic_text_summarizer.summarize(title, content)
        summary_produced = result['summarization']

        # print('summary_produced-------------\n', summary_produced)
        # print('-------------------------------')
        # print(sorted(result['C'].items(), reverse=True))

        score = evaluation.rouge_score(summary, summary_produced)

        precision1, recall1, fmeasure1 = evaluation.evaluation_results('rouge1', score)
        precision2, recall2, fmeasure2 = evaluation.evaluation_results('rouge2', score)
        precision3, recall3, fmeasure3 = evaluation.evaluation_results('rougeL', score)
        evaluation1['precision'].append(precision1)
        evaluation1['recall'].append(recall1)
        evaluation1['fmeasure'].append(fmeasure1)
        evaluation2['precision'].append(precision2)
        evaluation2['recall'].append(recall2)
        evaluation2['fmeasure'].append(fmeasure2)
        evaluation3['precision'].append(precision3)
        evaluation3['recall'].append(recall3)
        evaluation3['fmeasure'].append(fmeasure3)
    # print(max(evaluation1), min(evaluation1), np.average(evaluation1))
    # print(max(evaluation1),min(evaluation1),np.average(evaluation1))
    print(evaluation.print_stat(evaluation1,'precision','fmeasure'))
    print(evaluation.print_stat(evaluation2,'precision','fmeasure'))
    print(evaluation.print_stat(evaluation3,'precision','fmeasure'))
    print('done')


