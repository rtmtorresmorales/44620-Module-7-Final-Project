#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## CSIS 44620 Web Mining and Applied Natural Language Processing
## homework for Module 7  Final Project for Bonus
## Presented by Ramon Torres
## DEC 7, 2022
## https://github.com/rtmtorresmorales/44620-Module-7-Final-Project


# In[29]:


# 1 Code to extract article, orginial article was discarded because text was not available.  
import requests
from bs4 import BeautifulSoup
import pickle
import requests
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

from bs4 import BeautifulSoup
response = requests.get('https://web.archive.org/web/20210326110857/https://hackaday.com/2021/03/23/hey-google-is-my-heart-still-beating/')
print(response.status_code)
print(response.headers['content-type'])
parser = 'html.parser'
soup = BeautifulSoup(response.text, parser)
article_page = requests.get('https://web.archive.org/web/20210326110857/https://hackaday.com/2021/03/23/hey-google-is-my-heart-still-beating/')
article_html = article_page.text

with open('python-match.pkl', 'wb') as f:
    pickle.dump(article_page.text, f)

for header in soup.findAll('h1'):
    print('h1 header:', header)
    print('h1 text:', header.text)


# In[44]:


#2  Read and print article and polarity
with open('python-match.pkl', 'rb') as f:
    article_html = pickle.load(f)

parser = 'html.parser'
soup = BeautifulSoup(article_html, parser)
article_element = soup.find('article')
print(article_element.get_text())

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')
doc = nlp(article_element.get_text())
#docx = nlp(poem)
print ('Polarity: ', doc._.polarity)
# For added bonus print subjectivity
print ('Subjectivity: ', doc._.subjectivity)


# In[ ]:


# Polarity with .145 shows a timid positive sentiment, subjectivity with .514 is in the middle of author's opinion and a possible factual information.


# In[47]:


#3 Load the article text into a trained spaCy pipeline, and determine the 5 most frequent tokens
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from collections import Counter

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')
doc = nlp(article_element.get_text())

def we_care_about(token):
    return not (token.is_space or token.is_punct or token.is_stop)

interesting_tokens = [token for token in doc if we_care_about(token)]
word_freq = Counter(map(str,interesting_tokens))
print(word_freq.most_common(5))


# In[49]:


#4 Load the article text into a trained spaCy pipeline, and determine the 5 most frequent lemmas
interesting_lemmas = [token.lemma_ for token in doc if we_care_about(token)]
lemma_freq = Counter(interesting_lemmas)
print(lemma_freq.most_common(5))


# In[53]:


#4 Make a list containing the scores (using tokens) of every sentence in the article, and plot a histogram
interesting_token = list()
for token, freq in word_freq.most_common(5):
    interesting_token.append(token)

interesting_lemma = set()
for lemma, freq in lemma_freq.most_common(5):
    interesting_lemma.add(lemma)

sentences = list(doc.sents)
stringlist = list()
nmwrds = list()
for sentence in sentences:
    scount = 0
    sent_str = str(sentence).replace('\n','').replace('  ',' ')
    stringlist.append(sent_str)
    for token in sentence:
        if not(token.is_space or token.is_punct):
            scount +=1
    nmwrds.append(scount)

def score_sentence_by_token(sentence, interesting_token):
    tcount = 0
    for token in sentences[sentence]:
        if token.text.lower() in interesting_token:
            tcount += 1
    if tcount == 0:
        print('No interesting tokens')
    tscore = tcount/nmwrds[sentence]
    print('sentence:',stringlist[sentence], 'tokens:',tcount,'words:',nmwrds[sentence],'score:',tscore)

def score_sentence_by_lemma(sentence, interesting_lemma):
    lcount = 0
    for token in sentences[1]:
        if token.lemma_.lower() in interesting_lemma:
            lcount += 1
    if lcount == 0:
        print('No interesting lemmas')
    lscore = lcount/nmwrds[sentence]
    print('sentence:',stringlist[sentence], 'lemmas:',lcount,'words:',nmwrds[sentence],'score:',lscore)
    
score_sentence_by_token(1, interesting_token)
score_sentence_by_lemma(1, interesting_lemma)


# In[58]:


import matplotlib.pyplot as plt
import numpy as np
nmtkns = list()
tscores = list()
for sentence in sentences:
    tcount = 0
    for token in sentence:
        if token.text.lower() in interesting_token:
            tcount +=1
    nmtkns.append(tcount)
for i in nmwrds:
    tscores = [t/w for t,w in zip(nmtkns,nmwrds)]

t = np.array(tscores)  
plt.hist(t)
plt.xlabel('Scores')
plt.ylabel('Frequency')
plt.title('Tokens Histogram')
plt.xticks(np.arange(0, 1, 0.1))
plt.show()


# # Most common range from 0.0 to 0.1

# In[59]:


# 6 Make a list containing the scores (using lemmas) of every sentence in the article, and plot a histogram with appropriate  titles and axis labels of the scores.


import matplotlib.pyplot as plt
import numpy as np

nmlmas = list()
lscores = list()
for sentence in sentences:
    lcount = 0
    for token in sentence:
        if token.lemma_.lower() in interesting_lemma:
            lcount +=1
    nmlmas.append(lcount)
for i in nmlmas:
    lscores = [t/w for t,w in zip(nmlmas,nmwrds)]

l = np.array(lscores)  
plt.hist(l)
plt.xlabel('Scores')
plt.ylabel('Frequency')
plt.title('Lemmas Histogram')
plt.xticks(np.arange(0, 1, 0.1))
plt.show()


# # Most common range 0.0 to 0.1

# In[ ]:


# 7 Using the histograms from questions 5 and 6, decide a "cutoff" score for tokens and lemmas such that fewer than half the sentences would have a score greater than the cutoff score. Record the scores in this Markdown cell


# In[60]:



top_idx_t = np.argsort(x)[-10:]
top_values_t = [t[i] for i in top_idx_t]
print(min(top_values_t))

top_idx_l = np.argsort(x)[-10:]
top_values_l = [l[i] for i in top_idx_l]
print(min(top_values_l))


# # Cutoff Score (tokens): 0.058823529411764705
# # Cutoff Score (lemmas): 0.08571428571428572

# In[69]:


#8 Create a summary of the article by going through every sentence in the article and adding it to an (initially) empty list if its score (

tsummary = list()
tsumscores = list()

for sentence in sentences:
    tcount = 0
    scount = 0
    for token in sentence:
        if not(token.is_space or token.is_punct):
            scount += 1
        if token.text.lower() in interesting_token:
            tcount += 1
    if tcount != 0:
        tscore = tcount/scount
        if tscore >= min(top_values_l):
            sent_str = str(sentence).replace('\n','').replace('  ',' ')
            tsummary.append(sent_str)
            tsumscores.append(tscore)

print(' '.join(tsummary))


# In[70]:


# 9 Print the polarity score of your summary you generated with the token scores 
doc = nlp(' '.join(tsummary))

print("Token summary polarity: ", doc._.blob.polarity)
print(tsumscores)
print("Token summary sentence count: ", len(tsummary))


# In[68]:


# 10 Create a summary of the article by going through every sentence in the article 

lsummary = list()
lsumscores = list()

for sentence in sentences:
    lcount = 0
    scount = 0
    for token in sentence:
        if not(token.is_space or token.is_punct):
            scount += 1
        if token.lemma_.lower() in interesting_lemma:
            lcount += 1
    if lcount != 0:
        lscore = lcount/scount
        if lscore >= min(top_values_l):
            sent_str = str(sentence).replace('\n','').replace('  ',' ')
            lsummary.append(sent_str)
            lsumscores.append(lscore)

print(' '.join(lsummary))


# In[71]:


#11 Print the polarity score of your summary you generated


doc = nlp(' '.join(lsummary))

print("Lemma summary polarity: ", doc._.blob.polarity)
print(lsumscores)
print("Lemma summary sentence count: ", len(lsummary))


# In[ ]:


#12 Compare your polarity scores of your summaries to the polarity scores of the initial article. Is there a difference? Why do you think that may or may not be?. Answer in this Markdown cell.


# # 12 Polarity of tokenes and lemma summaries are less than orginal article,  original articles reached .145 versus lemma and token are in the range of .098 and .097.  Values are so low that may not be significant, but as we reduced the number od sentences and tokenize polarity keep moving to the center values of the actual polarity range between -1 and =+1.
# 

# In[ ]:


#13 Based on your reading of the original article, which summary do you think is better (if there's a difference). Why do you think this might be?


# #13 The summaries are pretty much the same, perhaps for reserach purposes a more detailed summary will be more beneficial to determine the applicability and usefulness of the article for a researcher.
