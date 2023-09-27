# -*- coding: utf-8 -*-
"""

@author: arsha
"""

# Importing the data file

import pandas as pd

df=pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/AS 11 Text mining data files/Elon_musk.csv", encoding='latin-1')
df.drop(columns='Unnamed: 0',inplace=True)
df
X=df['Text']


# Text preprocessing

df['Text'] = df.Text.map(lambda x : x.lower())
df['Text']


# Remove both the leading and the trailing characters

df=[Text.strip() for Text in df.Text] 


# Removes empty strings, because they are considered in Python as False

df=[Text for Text in df if Text]
df[0:10]


# Joining the list into one string/text

text = ' '.join(df)
text
type(text)
len(text)


#                   Generate wordcloud 

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
# Define a function to plot word cloud
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(15, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");


# Generate wordcloud

type(STOPWORDS)
stopwords = STOPWORDS
len(stopwords)
text
type(text)

wordcloud = WordCloud(width = 3000, height = 2000, background_color='black', max_words=100,colormap='Set2',stopwords=stopwords).generate(text)
plot_cloud(wordcloud)


#  Text preprocessing

import string         #  special operations on strings
no_punc_text = text.translate(str.maketrans('', '', string.punctuation)) 
no_punc_text


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Tokenization

from nltk.tokenize import word_tokenize
text_tokens = word_tokenize(no_punc_text)
len(text_tokens)
print(text_tokens[0:50])


# Remove stopwords

from nltk.corpus import stopwords


nltk.download('punkt')

nltk.download('stopwords')
my_stop_words = stopwords.words('english')

no_stop_tokens = [word for word in text_tokens if not word in my_stop_words]
len(no_stop_tokens)
print(no_stop_tokens[0:40])


# Joining the words in to single document

doc = ' '.join(my_stop_words)
doc
print(doc[0:40])


# Lemmatization

from nltk.stem import WordNetLemmatizer
Lemmatizer = WordNetLemmatizer()

lemmas = []
for token in doc.split():
    lemmas.append(Lemmatizer.lemmatize(token))

print(lemmas)
type(lemmas)

# Text preprocessing end 


#  Feature extraction 
# How we converted in features

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(lemmas)
X


# Every word and its position in the X

pd.DataFrame.from_records([vectorizer.vocabulary_]).T.sort_values(0,ascending=True).head(30)
pd.DataFrame.from_records([vectorizer.vocabulary_]).T.sort_values(0,ascending=True)
print(vectorizer.vocabulary_)
print(vectorizer.get_feature_names_out()[0:11])
print(vectorizer.get_feature_names_out()[50:100])
print(X.toarray()[50:100])


# feature extraction end 


# identifying combination of words, bigram,s trigrams
                # Let's see how can bigrams and trigrams can be included here
                             # Bigram
                             
                             
vectorizer_ngram_range = CountVectorizer(analyzer='word',ngram_range=(1,1),max_features = 120)
bow_matrix_ngram =vectorizer_ngram_range.fit_transform(df)
bow_matrix_ngram
type(df)

print(vectorizer_ngram_range.get_feature_names_out())
print(bow_matrix_ngram.toarray())

print(vectorizer_ngram_range.get_feature_names_out())
w1 = list(vectorizer_ngram_range.get_feature_names_out())
type(w1)
w2 = ' '.join(w1)
w2
type(w2)


# Plot

wordcloud = WordCloud(width = 3000, height = 2000, background_color='black', max_words=120,colormap='Set2',stopwords=stopwords).generate(w2)
plot_cloud(wordcloud)


# Trigram

vectorizer_ngram_range = CountVectorizer(analyzer='word',ngram_range=(2,2),max_features = 100)
bow_matrix_ngram =vectorizer_ngram_range.fit_transform(df)
bow_matrix_ngram

print(vectorizer_ngram_range.get_feature_names_out())
print(bow_matrix_ngram.toarray())

w3 = list(vectorizer_ngram_range.get_feature_names_out())
w3
w4 = ' '.join(w3)
w4


# Plot

wordcloud = WordCloud(width = 3000, height = 2000, background_color='black', max_words=100,colormap='Set2',stopwords=stopwords).generate(w4)
plot_cloud(wordcloud)



# Sentiment Analysis
# Importing the data file

df=pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/AS 11 Text mining data files/Elon_musk.csv", encoding='latin-1')
df.drop(columns='Unnamed: 0',inplace=True)
df
X=df['Text']
analyzer = SentimentIntensityAnalyzer()
df['Sentiment'] = df['Text'].apply(lambda Text: analyzer.polarity_scores(Text)['compound'])


#  Analyze Results

avg_sentiment = df['Sentiment'].mean()
positive_tweets = df[df['Sentiment'] > 0]
negative_tweets = df[df['Sentiment'] < 0]
neutral_tweets = df[df['Sentiment'] == 0]

print("Average Sentiment Score:", avg_sentiment)
print("Number of Positive Tweets:", len(positive_tweets))
print("Number of Negative Tweets:", len(negative_tweets))
print("Number of Neutral Tweets:", len(neutral_tweets))


















