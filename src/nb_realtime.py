import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

filename = 'nb_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

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

def predict_nb(text):
    text = preprocess(text)
    print(text)
    text = np.array([text])
    print(text)
    transformer = TfidfTransformer()
    tfv_loaded = TfidfVectorizer(sublinear_tf=True, stop_words="english",
                                 vocabulary=pickle.load(open("nb_feature.pkl", "rb")))
    text = transformer.fit_transform(tfv_loaded.fit_transform(text))
    print(text)
    polarity = loaded_model.predict(text)
    if polarity == 1:
        return 'POSITIVE'
    else:
        return 'NEGATIVE'

#text = input("Enter text whose polarity has to be calculated:")
#predict_nb(text)
