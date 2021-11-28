# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 14:49:14 2021

@author: harim
"""
import re
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
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
    with open("Raw" + c + ".txt", "r", encoding="utf8") as file:
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


unique_list = []
for comedian in data.columns:
    uniques = data[comedian].to_numpy().nonzero()[0].size
    unique_list.append(uniques)

# Create a new dataframe that contains this unique word count
data_words = pd.DataFrame(list(zip(comediantes, unique_list)), columns=['comedian', 'unique_words'])
data_unique_sort = data_words.sort_values(by='unique_words')

total_list = []
for comedian in data.columns:
    print(comedian)
    totals = sum(data[comedian])
    total_list.append(totals)


# Comedy special run times from IMDB, in minutes
run_times = [9.5, 10, 12.5, 10, 10.5, 13]

# Let's add some columns to our dataframe
data_words['total_words'] = total_list
data_words['run_times'] = run_times
data_words['words_per_minute'] = data_words['total_words'] / data_words['run_times']

# Sort the dataframe by words per minute to see who talks the slowest and fastest
data_wpm_sort = data_words.sort_values(by='words_per_minute')

print(data_wpm_sort)

y_pos = np.arange(len(data_words))

plt.subplot(1, 2, 1)
plt.barh(y_pos, data_unique_sort.unique_words, align='center')
plt.yticks(y_pos, data_unique_sort.comedian)
plt.title('Número de palabras únicas', fontsize=20)

plt.subplot(1, 2, 2)
plt.barh(y_pos, data_wpm_sort.words_per_minute, align='center')
plt.yticks(y_pos, data_wpm_sort.comedian)
plt.title('Número de palabras por minuto', fontsize=20)

plt.tight_layout()
plt.show()

Counter(words).most_common()

asd = data.transpose()[['pendejo', 'pendeja', 'pendejada', 'pinche','pinches','putos','puta','puto','putazo',
                                   'chingada','chinga','culero','culera','cabrón','cabrones','cabrona',
                                   'güey','huevo','verga','madres','mames','mamada','mamón']]
data_profanity = pd.concat([asd.pendejo + asd.pendeja + asd.pendejada + asd.pinche + asd.pinches + asd.putos + 
                            asd.puta + asd.puto + asd.putazo + asd.chingada + asd.chinga  + asd.culero + asd.culera + asd.cabrón + 
                            asd.cabrones + asd.cabrona, 
                            asd.güey + asd.huevo + asd.verga + asd.madres + asd.mames + asd.mamada + asd.mamón, 
                            ], axis=1)
data_profanity.columns = ['groserías_P_y_C','Otras_groserías']
print(data_profanity)

plt.rcParams['figure.figsize'] = [10, 8]

for i, comedian in enumerate(data_profanity.index):
    x = data_profanity.groserías_P_y_C.loc[comedian]
    y = data_profanity.Otras_groserías.loc[comedian]
    plt.scatter(x, y, color='blue')
    plt.text(x+0.5, y+0.5, comediantes[i], fontsize=10)
    plt.xlim(0, 30) 
    
plt.title('Número de palabras malsonantes utilizadas', fontsize=20)
plt.xlabel('Grupo A', fontsize=15)
plt.ylabel('Grupo B', fontsize=15)

plt.show()
