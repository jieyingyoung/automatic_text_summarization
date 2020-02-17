#!/usr/bin/env python
# coding: utf-8


import re
import jieba
import numpy as np
from sklearn.decomposition import PCA
from gensim.models import Word2Vec


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
        
        all_sentence_embeddings = self.compute_sentence_embeddings([title, content] + sentences)
        Vt = all_sentence_embeddings[0]
        Vc = all_sentence_embeddings[1]
        Vs = all_sentence_embeddings[2:]
        
        C = self.compute_C(Vs, Vt, Vc)
        
        summarize_sentences_size = int(self.summarize_sentences_size * len(sentences))
        summarize_sentences = [(sentence, index) for sentence, index, _ in sorted(zip(sentences, range(len(sentences)), C), key=lambda e: e[2], reverse=True)[:summarize_sentences_size]]
        summarization = ''.join([sentence for sentence, _ in sorted(summarize_sentences, key=lambda e: e[1])])
        
        return {
            'summarization': summarization,
            'C': dict(zip(C,sentences)),
            'Vs': dict(zip(sentences,Vs)),
            'Vc': Vc,
            'Vt': Vt
        }
    
    def compute_sentence_embeddings(self, sentences):
        model = self.word_vec_model
        a = self.a
        sentence_embeddings = np.array([self.compute_sentence_embedding(sentence, model, a) for sentence in sentences])
        u = self.compute_first_principal_component(sentence_embeddings)
        return (1 - np.dot(u, u.T)) * sentence_embeddings
    
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
        si_t_cs = abs(np.sum(Vsi * Vt) / (np.sqrt(np.sum(Vsi**2)) * np.sqrt(np.sum(Vt**2))))

        si_c_cs = abs(np.sum(Vsi * Vc) / (np.sqrt(np.sum(Vsi**2)) * np.sqrt(np.sum(Vc**2))))

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
        word_embeddings = np.array([a / (a + (model.wv.vocab[word].count / model.corpus_total_words)) * model.wv[word] for word in words if word in model.wv])
        return np.sum(word_embeddings, axis=0) / word_embeddings.shape[0]
    
    @staticmethod
    def compute_first_principal_component(sentence_embeddings):
        pca = PCA(n_components=1)
        pca.fit(sentence_embeddings)
        return pca.components_
    
    @staticmethod
    def cut_words(content: str):
        return [word for word in list(jieba.cut(AutomaticTextSummarizer.clean_data(content))) if word != ' ']
    
    @staticmethod
    def cut_sentences(content: str):
        sentence_division = '[〇一-\u9fff㐀-\u4dbf豈-\ufaff𠀀-\U0002a6df𪜀-\U0002b73f𫝀-\U0002b81f丽-\U0002fa1f⼀-⿕⺀-⻳0-9a-zA-G ＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·]*[！？｡。][」﹂”』’》）］｝〕〗〙〛〉】]*'
        return re.findall(sentence_division, content)
    
    @staticmethod
    def clean_data(content: str):
        chinese_punctuation = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'
        english_punctuation = ',.?;:\'"`~!\[\]\-'
        special_char = r'<>/\\|\[\]{}@#\$%\^&\*\(\)-\+=_\n'
        return re.sub('(?P<punctuation>[{}]|[{}])|(?P<special_char>[{}])'.format(chinese_punctuation, english_punctuation, special_char), ' ', content)


if __name__ == '__main__':

    title = """全面贯彻新时代军事教育方针
    一论认真学习贯彻习主席在全军院校长集训开班式上的重要讲话"""
    content = """治军先治校，强军必强校。在强军事业对人才提出强劲需求、新时代院校体系重塑后转型升级的关节点上，习主席出席全军院校长集训开班式并发表重要讲话，体现了对军队院校建设和人才培养的高度重视，对全军教育战线广大官兵的关心厚爱。习主席站在时代发展和战略全局高度，回答了院校建设和人才培养带根本性、方向性的一系列重大问题，鲜明提出新时代军事教育方针，对全面深化军事院校改革、提高院校长办学治校能力作出重大部署，为开创院校教育和人才培养新局面提供了科学指南和根本遵循。这必将开启军事教育新的历史征程，汇聚起人才强军的磅礴力量。

    强军兴军，关键靠人才，基础在教育。院校教育是我军人才培养的主渠道，具有基础性、先导性、全局性作用。我党我军对办学育人历来高度重视。我军之所以能够不断发展壮大，完成党在各个历史时期赋予的使命任务，一个很重要的原因就是重视人才培养。人才强则事业强，人才兴则军队兴。当前，世界军事领域围绕人才和科技的竞争日趋激烈，我国安全形势正在发生新的深刻变化，我军职能任务不断拓展，我军建设正加快向质量效能型和科技密集型转变，这对我们培养军事人才和办好军事教育提出更高要求。

    发展军事教育，必须有一个管总的方针，解决好培养什么人、怎样培养人、为谁培养人这个根本问题。习主席在讲话中指出，新时代军事教育方针，就是坚持党对军队的绝对领导，为强国兴军服务，立德树人，为战育人，培养德才兼备的高素质、专业化新型军事人才。这一军事教育方针，着眼院校建设和人才培养的长远大计，赋予了军事教育鲜明的时代要求和强军指向，是做好军事教育工作的基本遵循，标志着我党我军对军事教育规律和军事人才培养规律的认识提升到新的境界。

    贯彻新时代军事教育方针，关系新时代军事教育和人才培养的方向与全局。要坚持正确政治方向，以习近平新时代中国特色社会主义思想为指导，贯彻习近平强军思想，贯彻新时代军事战略方针，把政治建军要求贯彻到军事教育全部实践中，确保军事教育领域始终成为坚持党的领导的坚强阵地。坚持立德树人，把思想政治教育贯穿育人全过程，确保枪杆子永远掌握在忠于党的、可靠的人手中，确保党和军队事业后继有人。坚持为战育人，打仗需要什么就教什么，部队需要什么就练什么，确保培养的人才能够打赢现代战争。坚持一体化布局，推进联合育人、开放育人、全程育人，形成高水平军事人才培养体系。坚持内涵式发展，厚植发展基础，增强发展活力，推动军事教育事业高质量发展。

    全军各级要认真学习领会习主席重要讲话精神，深刻领会丰富内涵和精神实质。要毫不动摇贯彻落实新时代军事教育方针，并结合新的实践不断丰富发展。要全面实施人才强军战略，全面深化军事院校改革创新，把培养人才摆在更加突出的位置，培养德才兼备的高素质、专业化新型军事人才，努力在新的起点推动院校教育和人才培养迈上新台阶。"""


    path = '../model/word2vec_normal_full.model'
    model = load_model(path)

    automatic_text_summarizer = AutomaticTextSummarizer(model)
    automatic_text_summarizer.set_n_neighbors(2)

    automatic_text_summarizer.set_summarize_sentences_size(0.4)

    result = automatic_text_summarizer.summarize(title, content)

    print(result['summarization'])
    print('-------------------------------')
    print(sorted(result['C'].items(),reverse = True))

