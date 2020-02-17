import gensim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def similarity(word):
    similar_words = model.most_similar(word)
    print('{}的相似词有：'.format(word),similar_words)
    return similar_words

def analogy(x1,x2,y1):
    result = model.most_similar(positive=[y1,x2],negative=[x1])
    print('{} + {} - {} 的词是：'.format(y1,x2,x1),result[0][0])
    return result[0][0]


def tsne_plot(model):
    # Creates and TSNE model and plots it"

    labels = []
    tokens = []
    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    # to display Chinese when networkx is used
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.show()

if __name__ == '__main__':
    MODEL_PATH = "../model/word2vec_normal_full.model"
    model = gensim.models.Word2Vec.load(MODEL_PATH)

    # test1-similarity
    similar_words =similarity('爱')

    # # test2-analogy
    analogy_result = analogy('中国','汉语','美国')
    analogy_result2 = analogy('男人','皇帝','女人')

    # vector visualization
    # tsne_plot(model)

    # MODEL_SELECTIVE_PATH = "../model/word2vec_selective_10000.model"
    # model_selevtive = gensim.models.Word2Vec.load(MODEL_SELECTIVE_PATH)
    # tsne_plot(model_selevtive)

