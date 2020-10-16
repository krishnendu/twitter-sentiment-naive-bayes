import time
time1=time.time()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import re
from string import punctuation

data = pd.read_csv('C:/Users/Kris/Desktop/Projects/Twitter Sentiment Analysis/finalSentimentdata2.csv',sep=',')
data.head()

sentiment_map={'anger':-2,'fear':-1,'sad':1,'joy':2}
data.insert(2,'sentiment_int',[sentiment_map[s] for s in data.sentiment],True)

# data.head()

# data.info()


# data.sentiment_int.value_counts()


# sentiment_count = data.groupby('sentiment_int').count()
# plt.bar(sentiment_count.index.values,sentiment_count['text'])
# plt.xlabel('Review sentiment')
# plt.ylabel('No. of review')
# plt.show()

# from sklearn.feature_extraction.text import CountVectorizer
# from nltk.tokenize import RegexpTokenizer

def preProcessor(text):
    text=re.sub(r'(http|ftp|https):\/\/([\w\-_]+(?:(?:\.[\w\-_]+)+))([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?', ' ', text)
    text=re.sub(r'['+punctuation+']',' ',text)
    return text

token=RegexpTokenizer(r'\w+')
cv=CountVectorizer(lowercase=True,preprocessor=preProcessor,stop_words='english',ngram_range=(1,1),tokenizer=token.tokenize)
text_counts=cv.fit_transform(data['text'])

# from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(text_counts,data['sentiment_int'],test_size=0.3)

# from sklearn.naive_bayes import MultinomialNB
# from sklearn import metrics

clf=MultinomialNB()
clf.fit(x_train,y_train)
pred=clf.predict(x_test)
accuracy=metrics.accuracy_score(y_test, pred)

print("Accuracy is : ",(accuracy*100),"%")
time2=time.time()
print("Time required : ",time2-time1)

