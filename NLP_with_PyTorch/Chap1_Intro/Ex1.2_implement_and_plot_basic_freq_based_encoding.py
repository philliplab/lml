from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import IPython
ipython = IPython.get_ipython()
ipython.magic('matplotlib')

corpus = ['Time flies like an arrow', 'Fruit flies like a banana']

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus).toarray()
vocab = sorted(tfidf_vectorizer.vocabulary_.keys())

sns.heatmap(tfidf, cbar = False, annot = True, xticklabels = vocab)
