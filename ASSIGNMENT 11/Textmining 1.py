# -*- coding: utf-8 -*-
"""

@author: arsha
"""



# Importing the data file

import pandas as pd
df=pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/AS 11 Text mining data files/amazon.csv")
df
pd.set_option('display.max_columns', None)
df['reviewerName']
df['reviewText']
df.drop(columns='Unnamed: 0',inplace=True)
df
X=df['reviewText']
X
df.info()


###########################################################################


import re

def remove_float_values(reviewText):
    # Ensure 'text' is a string
    if not isinstance(reviewText, str):
        return reviewText  # If it's not a string, return it as is (pass-through)

    # Regular expression pattern to match float values
    float_pattern = r'\b\d+\.\d+\b|\b\d+\b'

    # Use the sub() method to replace float values with an empty string
    cleaned_text = re.sub(float_pattern, '', reviewText)

    return cleaned_text
df['reviewText'] = df['reviewText'].apply(remove_float_values)



# remove both the leading and the trailing characters

def strip_text(reviewText):

    # Check if the value is a string before applying the strip() method
    return reviewText.strip() if isinstance(reviewText, str) else reviewText

# Apply the strip_text function to the 'reviewText' column
df['reviewText'] = df['reviewText'].apply(strip_text)



#removes empty strings, because they are considered in Python as False

df=[Text for Text in df if Text]


df=[reviewText for reviewText in df if reviewText]
df[0:10]
df
# Joining the list into one string/text
text = ' '.join(df)
text
type(text)
len(text)


#          Generate Wordcloud

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

# Plot
wordcloud = WordCloud(width = 3000, height = 2000, background_color='black', max_words=100,colormap='Set2',stopwords=stopwords).generate(text)
plot_cloud(wordcloud)


# text preprocessing start 
# Punctuation

import string # special operations on strings
no_punc_text = text.translate(str.maketrans('', '', string.punctuation)) 
no_punc_text


# Tokenization

from nltk.tokenize import word_tokenize
text_tokens = word_tokenize(no_punc_text)
len(text_tokens)
print(text_tokens[0:50])


# Remove stopwords

import nltk
from nltk.corpus import stopwords


# nltk.download('punkt')

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


#    Text preprocessing end 



#                     Feature extraction 

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(lemmas)
X


# every word and its position in the X

pd.DataFrame.from_records([vectorizer.vocabulary_]).T.sort_values(0,ascending=True).head(30)
pd.DataFrame.from_records([vectorizer.vocabulary_]).T.sort_values(0,ascending=True)
print(vectorizer.vocabulary_)
print(vectorizer.get_feature_names_out()[0:11])
print(vectorizer.get_feature_names_out()[50:100])
print(X.toarray()[50:100])

#                     Feature extraction end 


#  Identifying combination of words, bigram,s trigrams
# Let us see how the bigrams and trigrams can be included here
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


#  Trigram

vectorizer_ngram_range = CountVectorizer(analyzer='word',ngram_range=(1,2),max_features = 100)
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


# Emotion Minning

import pandas as pd
df=pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/AS 11 Text mining data files/amazon.csv")
df
from textblob import TextBlob
def get_sentiment(reviewText):
    # Create a TextBlob object
    blob = TextBlob(reviewText)
    
    # Get sentiment polarity (-1 to 1, where -1 is negative, 0 is neutral, and 1 is positive)
    sentiment_polarity = blob.sentiment.polarity
    
    # Map polarity to emotion
    if sentiment_polarity > 0:
        emotion = 'Positive'
    elif sentiment_polarity < 0:
        emotion = 'Negative'
    else:
        emotion = 'Neutral'
        
    return emotion

# Example usage:
text = df['reviewText']
sentiment = get_sentiment(text)
print("Sentiment:", sentiment)





