import gensim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def similarity(word):
    similar_words = model.most_similar(word)
    print('The similar words with ""{}"" are ：\n'.format(word),similar_words)
    return similar_words

def analogy(a,b,c):
    result = model.most_similar(positive=[a,c],negative=[b])
    # print('{} - {} + {} = ：'.format(a,b,c),result[0][0])
    print('If {} is a word for {}, what is the word for {} ? --> '.format(a,b,c),result[0][0])
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

    plt.figure(figsize=(160, 160))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig()
    plt.show()
    

if __name__ == '__main__':
    
    MODEL_PATH = "../model/word2vec_enfull.model"
    model = gensim.models.Word2Vec.load(MODEL_PATH)

    # test1-similarity
    # similar_words =similarity('love')

    # test2-analogy
    analogy_result = analogy('fire','water','girl')
    analogy_result2 = analogy('woman','man', 'King' )
    analogy_result3 = analogy('sun','moon', 'banana' )
    analogy_result4 = analogy('strong','hard', 'clever' )
    # vector visualization
    # tsne_plot(model) 

    MODEL_SELECTIVE_PATH = "../model/word2vec_draw_en.model"
    model_selevtive = gensim.models.Word2Vec.load(MODEL_SELECTIVE_PATH)
    tsne_plot(model_selevtive)

