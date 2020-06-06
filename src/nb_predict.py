import pickle
import pandas as pd
import numpy as np
import re
import os

filename = 'nb_model.sav'

file_path = os.path.join('IMDB Dataset.csv')
data = pd.read_csv(file_path)
print(data.head())

y = data.sentiment
y.head()

label = {'positive':1, 'negative':-1}

def preprocess_y(sentiment):
    return label[sentiment]

y = y.apply(preprocess_y)
y.head()

X = data.review
X.head()

import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

import re
def preprocess(review):
    #convert the tweet to lower case
    review.lower()
    #convert all urls to sting "URL"
    review = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',review)
    #convert all @username to "AT_USER"
    review = re.sub('@[^\s]+','AT_USER', review)
    #correct all multiple white spaces to a single white space
    review = re.sub('[\s]+', ' ', review)
    #convert "#topic" to just "topic"
    review = re.sub(r'#([^\s]+)', r'\1', review)
    tokens = word_tokenize(review)
    tokens = [w for w in tokens if not w in stop_words]
    return " ".join(tokens)

X = X.apply(preprocess)
print(X.head())

from sklearn.feature_extraction.text import TfidfVectorizer
def feature_extraction(data):
    tfv = TfidfVectorizer(sublinear_tf=True, stop_words = "english")
    features = tfv.fit_transform(data)
    pickle.dump(tfv.vocabulary_, open("nb_feature.pkl", "wb"))
    return features

data = np.array(X)
label = np.array(y)
features = feature_extraction(data)
print(features)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.20)


loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)

text = input("Enter text whose polarity has to be calculated:")
#text = data.review[3]
text = preprocess(text)
print(text)
text = np.array([text])
print(text)

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
tfv_loaded = TfidfVectorizer(sublinear_tf=True, stop_words = "english", vocabulary=pickle.load(open("nb_feature.pkl", "rb")))
text = transformer.fit_transform(tfv_loaded.fit_transform(text))
print(text)
polarity = loaded_model.predict(text)
print(polarity)