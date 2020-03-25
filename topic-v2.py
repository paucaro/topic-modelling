# packages to store and manipulate data
import pandas as pd
# importing required libraries 
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
# Importing Gensim
import gensim
from gensim import corpora

df = pd.read_csv("data.csv")
doc_complete = df['command']
# print(doc_complete)

# set of stopwords
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete]    

#print('\n\nCleaned Data\n\n')
#print(doc_clean)

dictionary = corpora.Dictionary(doc_clean)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix, num_topics=7, id2word = dictionary, passes=50)
#print(ldamodel.print_topics(num_topics=7, num_words=4))


##### test
unseen_document = 'hello, how are you?'
bow_vector = dictionary.doc2bow(clean(unseen_document).split())

for index, score in sorted(ldamodel[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, ldamodel.print_topic(index, 4)))