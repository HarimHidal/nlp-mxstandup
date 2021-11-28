# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 20:02:57 2021

@author: harim
"""

import re
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import numpy as np
import math

def split_text(text, n=10):
    '''Takes in a string of text and splits into n equal parts, with a default of 10 equal parts.'''

    # Calculate length of text, the size of each chunk of text and the starting points of each chunk of text
    length = len(text)
    size = math.floor(length / n)
    start = np.arange(0, length, size)
    
    # Pull out equally sized pieces of text and put it into a list
    split_list = []
    for piece in range(n):
        split_list.append(text[start[piece]:start[piece]+size])
    return split_list

def combine_text(list_of_text):
    '''toma una lista de string y devuelve un sólo string'''
    combined_text = ' '.join(list_of_text)
    return combined_text

def clean_text_round1(text):
    '''minusculas, quita brakcets, puntuación, \n y palabras con números'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('\\n', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text

data = {}

comediantes = ["Richy", "Franco", "Carlos", "Sofía", "Alan", "Alex"]

for c in comediantes:
    with open("TRaw" + c + ".txt", "r", encoding="utf8") as file:
        data[c] = file.readlines()
        file.close
        
data_combined = combine_text(data)

data_combined = {key: [combine_text(value)] for (key, value) in data.items()}
#print(data_combined)

pd.set_option('max_colwidth',150)
data_df = pd.DataFrame.from_dict(data_combined).transpose()
data_df.columns = ['transcript']
data_df = data_df.sort_index()

#print(data_clean)

round1 = lambda x: clean_text_round1(x)
data_clean = pd.DataFrame(data_df.transcript.apply(round1))

#print(data_clean)

spanish = []
with open('Spanish.txt', "r", encoding="utf8") as file:
    spanish = file.read().splitlines()
    file.close()
    
cv = CountVectorizer(stop_words=spanish)
data_cv = cv.fit_transform(data_clean.transcript)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data_clean.index

#print(data_dtm)

data = data_dtm.transpose()

top_dict = {}
for c in data.columns:
    top = data[c].sort_values(ascending=False).head(30)
    top_dict[c]= list(zip(top.index, top.values))

#print(top_dict)

# Print the top 15 words said by each comedian
for comedian, top_words in top_dict.items():
    print(comedian)
    print(', '.join([word for word, count in top_words[0:14]]))
    print('---')
    
words = []
for comedian in data.columns:
    top = [word for (word, count) in top_dict[comedian]]
    for t in top:
        words.append(t)

#print(words)

Counter(words).most_common()
add_stop_words = [word for word, count in Counter(words).most_common() if count > 1]

#print(add_stop_words)

spanish = spanish + add_stop_words
cv = CountVectorizer(stop_words=spanish)
data_cv = cv.fit_transform(data_clean.transcript)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data_clean.index

#print(data_dtm)

comediantes = ["Alan Saldaña", "Álex Fernández", "Carlos Ballarta", "Franco Escamilla", "Ricardo O'Farrill", "Sofía Niño de Rivera"]

wc = WordCloud(stopwords=spanish, background_color="white", colormap="Dark2",               max_font_size=80, random_state=42)

plt.rcParams['figure.figsize'] = [16, 6]

for index, comedian in enumerate(data.columns):
    wc.generate(data_clean.transcript[comedian])
    plt.subplot(3, 4, index+1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(comediantes[index])
    
plt.show()

data = data_clean
data['full_name'] = comediantes

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

data['polarity'] = data['transcript'].apply(pol)
data['subjectivity'] = data['transcript'].apply(sub)

#print(data)

plt.rcParams['figure.figsize'] = [8, 8]

for index, comedian in enumerate(data.index):
    x = data.polarity.loc[comedian]
    y = data.subjectivity.loc[comedian]
    plt.scatter(x, y, color='blue')
    plt.text(x+.001, y+.001, data['full_name'][index], fontsize=10)
    plt.xlim(-.25, .25) 
    
plt.title('Análisis de Sentimiento', fontsize=20)
plt.xlabel('<-- Negativo -------- Positivo -->', fontsize=15)
plt.ylabel('<-- Hechos -------- Opiniones -->', fontsize=15)

plt.show()

list_pieces = []
for t in data.transcript:
    split = split_text(t)
    list_pieces.append(split)
    
polarity_transcript = []
for lp in list_pieces:
    polarity_piece = []
    for p in lp:
        polarity_piece.append(TextBlob(p).sentiment.polarity)
    polarity_transcript.append(polarity_piece)
    
print(polarity_transcript)

plt.rcParams['figure.figsize'] = [14, 10]

for index, comedian in enumerate(data.index):    
    plt.subplot(3, 4, index+1)
    plt.plot(polarity_transcript[index])
    plt.plot(np.arange(0,10), np.zeros(10))
    plt.title(data['full_name'][index])
    plt.ylim(ymin=-.9, ymax=.9)
    
plt.show()