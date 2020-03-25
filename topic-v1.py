import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import nltk
from pprint import pprint

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# print(lemmatize_stemming('referenced'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        result.append(lemmatize_stemming(token))
        #if token not in gensim.parsing.preprocessing.STOPWORDS:
        #    result.append(lemmatize_stemming(token))
    return result

if __name__ == '__main__':
    data = pd.read_csv("data.csv", error_bad_lines=False)
    data_text = data[['command']]
    data_text['index'] = data_text.index
    documents = data_text

#print(len(documents))
#print(documents[:5])

#DATA Pre-processing
# - Tokenization: split text into sentences and the sentences into words. Lowercase the words and 
# remove punctuation
# - stopwords removed
# - word are lemmatized - words in thrid person are changed to first person and verb in past
# and future tenses are changed into present
# - words are stemmed - words are reduced to their root form

    stemmer = PorterStemmer()

    processed_docs = documents['command'].map(preprocess)

    dictionary = gensim.corpora.Dictionary(processed_docs)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=4)
    #for idx, topic in lda_model.print_topics(-1):
    #    print('Topic: {} \nWords: {}'.format(idx, topic))
    
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
    #for idx, topic in lda_model_tfidf.print_topics(-1):
    #    print('Topic: {} Word: {}'.format(idx, topic))


    ##### test
    unseen_document = 'i want to know your color'
    bow_vector = dictionary.doc2bow(preprocess(unseen_document))

    for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):
        print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 5)))




    