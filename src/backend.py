from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def Analyze(msg):
    tb = TextBlob(msg)
    sid = SentimentIntensityAnalyzer()
    vad = sid.polarity_scores(msg)
    #print(tb.sentiment.polarity)
    #print(vad)
    return (tb.sentiment.polarity + vad['compound'])/2

#m = input("Enter text:")
#print("Sentiment polarity of given text is:"+ str(Analyze(m)))