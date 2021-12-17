import pandas as pd
import numpy as np
import nltk
import re
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

clean_data = pd.read_csv('csv/Tweets.csv')
#clean_data.head()

#clean_data.info()
sns.countplot(x = "airline_sentiment", data = clean_data)

# Önce ihtiyacımız olan sütunları bırakıyoruz

waste_col = ['tweet_id', 'airline_sentiment_confidence',
       'negativereason', 'negativereason_confidence', 'airline',
       'airline_sentiment_gold', 'name', 'negativereason_gold',
       'retweet_count', 'tweet_coord', 'tweet_created',
       'tweet_location', 'user_timezone']

data = clean_data.drop(waste_col, axis = 1)

#data.head()

#data.info()

def sentiment(x):
    if x == 'positive':
        return 1
    elif x == 'negative':
        return -1
    else:
        return 0


nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

stopwords = stopwords.words('english')
stemmer = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'\w+')
# As this dataset is fetched from twitter so it has lots of people tag in tweets
# we will remove them
tags = r"@\w*"


def preprocess_text(sentence, stem=False):
       sentence = [re.sub(tags, "", sentence)]
       text = []
       for word in sentence:

              if word not in stopwords:

                     if stem:
                            text.append(stemmer.stem(word).lower())
                     else:
                            text.append(word.lower())
       return tokenizer.tokenize(" ".join(text))


print(f"Orignal Text : {data.text[11]}")
print()
print(f"Preprocessed Text : {preprocess_text(data.text[11])}")


#this is an example vocabulary just to make concept clear
sample_vocab = ['the', 'cat', 'sat', 'on', 'mat', 'dog', 'run', 'green', 'tree']
# vocabulary of words present in dataset
data_vocab = []
for text in data.text:
    for word in text:
        if word not in data_vocab:
            data_vocab.append(word)

#function to return one-hot representation of passed text
def get_onehot_representation(text, vocab = data_vocab):
    onehot_encoded = []
    for word in text:
        temp = [0]*len(vocab)
        temp[vocab.index(word)-1] = 1
        onehot_encoded.append(temp)
    return onehot_encoded

print("One Hot Representation for sentence \"the cat sat on the mat\" :")
get_onehot_representation(['the', 'cat', 'sat', 'on', 'the', 'mat'], sample_vocab)

#data.text = data.text.map(preprocess_text)
#data.head()

print(f'Length of Vocabulary : {len(data_vocab)}')
print(f'Sample of Vocabulary : {data_vocab[120 : 142]}')
sample_one_hot_rep = get_onehot_representation(data.text[7], data_vocab)
print(f"Shapes of a single sentence : {np.array(sample_one_hot_rep).shape}")

#one-hot representation for dataset sentences

# data.loc[:, 'one_hot_rep'] = data.loc[:, 'text'].map(get_onehot_representation)

#if you run this cell it will give you a memory error

from sklearn.feature_extraction.text import CountVectorizer

sample_bow = CountVectorizer()

# sample_corpus = [['the', 'cat', 'sat'],
#                  ['the', 'cat', 'sat', 'in', 'the', 'hat'],
#                  ['the', 'cat', 'with', 'the', 'hat']]

sample_corpus = ["the cat sat", "the cat sat in the hat", "the cat with the hat"]

sample_bow.fit(sample_corpus)


def get_bow_representation(text):
    return sample_bow.transform(text)


print(f"Vocabulary mapping for given sample corpus : \n {sample_bow.vocabulary_}")
print("\nBag of word Representation of sentence 'the cat cat sat in the hat'")
print(get_bow_representation(["the cat cat sat in the hat"]).toarray())

sample_bow = CountVectorizer(binary=True)

sample_corpus = ["the cat sat", "the cat sat in the hat", "the cat with the hat"]

sample_bow.fit(sample_corpus)


def get_bow_representation(text):
    return sample_bow.transform(text)


print(f"Vacabulary mapping for given sample corpus : \n {sample_bow.vocabulary_}")
print("\nBag of word Representation of sentence 'the the the the cat cat sat in the hat'")
print(get_bow_representation(["the the the the cat cat sat in the hat"]).toarray())
#data.head()

bow = CountVectorizer()
bow_rep = bow.fit_transform(data.loc[:, 'text'].astype('str'))

print(f"Shape of Bag of word representaion matrix : {bow_rep.toarray().shape}")
