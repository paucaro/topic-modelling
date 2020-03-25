# Topic Modelling

Latent Dirichlet Allocation (LDA) is a algorithms used to discover the topics that are present in a corpus. Python with some libraries could be used to to find topics in text.

## How to install
Install pandas to load csv file:
``` 
pip install pandas
pip install gensim
pip install numpy
pip install nltk
```
## How to use
To run topic-v1:
```
python topic-v1.py
```
To run topic-v2:
```
python topic-v2.py
```
To run topic-v3:
```
python topic-v3.py
```
# Implementation
## Topic v1 and Topic v2
### DATA Pre-processing
- Tokenization: split text into sentences and the sentences into words. Lowercase the words and remove punctuation
- Stopwords removed
- Word are lemmatized - words in thrid person are changed to first person and verb in past and future tenses are changed into present
- Words are stemmed - words are reduced to their root form

Using gensim library to get topic modelling.

## Topic v3
NMF and LDA comparison with sklearn library

