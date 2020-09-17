from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import IPython
ipython = IPython.get_ipython()
ipython.magic('matplotlib')

corpus = ['Time flies like an arrow', 
          'Fruit flies like a banana']

one_hot_vectorizer = CountVectorizer(binary = True)
one_hot = one_hot_vectorizer.fit_transform(corpus).toarray()
vocab = sorted(one_hot_vectorizer.vocabulary_.keys())

sns.heatmap(one_hot, annot = True, cbar = False, xticklabels = vocab, yticklabels = ['Sentence 1', 'Sentence 2'])



