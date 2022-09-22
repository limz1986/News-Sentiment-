# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 13:51:03 2022

@author: 65904
"""
#%% Libraries

import numpy as np
import pandas as pd
from datetime import datetime 
import datetime as dt
import scipy.stats as scs
from scipy.stats.mstats import winsorize
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)     # Display all columns of statistics
sns.set_theme(style="white")
import yfinance as yf
from pandas_datareader import data as pdr
import re
import nltk
from gensim.models import Word2Vec


# Import stopwords with nltk.
from nltk.corpus import stopwords
stop = stopwords.words('english')

#%% SP500 Market Data Swope with ^DJI ^GSPC

#Market Returns
# Specify the starting and ending dates for the time series
start = datetime(2008, 1, 1)
end  = datetime(2016, 12, 31)

Stock_list = ['^GSPC']
SPY_dict = {}

for i in Stock_list:        
    locals()[i] = pdr.get_data_yahoo(i, start, end) 
    SPY_dict[i] =  locals()[i]['Adj Close']     
SP500_df = pd.DataFrame(SPY_dict)

# SP500_df.info()
SP500_df.reset_index(inplace=True)
SP500_df = SP500_df.rename(columns = {'^GSPC': 'Adj_Close'})

SP500_df=SP500_df[['Date','Adj_Close']]
SP500_df = SP500_df.rename(columns = {'Date': 'date'})
SP500_df.head()


# lets get forward returns

SP500_df.head(35)
SP500_df['close_30D']=SP500_df['Adj_Close'].shift(60)
SP500_df['ret_30D']  =SP500_df.Adj_Close/SP500_df.close_30D-1
SP500_df.dropna(inplace=True)
SP500_df.head(2)

SP500_df.describe()


#%% Text Processing

df=pd.read_csv(r'C:/Users/65904/Desktop/ML/NLP/Data.csv', encoding = "ISO-8859-1")
df2=pd.read_csv(r'C:/Users/65904/Desktop/ML/NLP/Combined_News_DJIA.csv', encoding = "ISO-8859-1")
df = df.drop(["Label"], axis=1)
df2 = df2.drop(["Label"], axis=1)
df = pd.merge(df, df2, how = "left", on = ['Date'])
df.dropna(inplace=True)

df_date= df.copy()
df_date.Date.min(),df_date.Date.max()


# Removing punctuations
data = df.iloc[:,1:51]
data.replace("[^a-zA-Z0-9]"," ",regex=True, inplace=True)

# Renaming column names for ease of access
list1= [i for i in range(50)]
new_Index=[str(i) for i in list1]
data.columns= new_Index
data.head(5)

# Convertng headlines to lower case
for index in new_Index:
    data[index]=data[index].str.lower()
data.head(1)

' '.join(str(x) for x in data.iloc[1,0:50])

headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:50]))
df = pd.DataFrame(headlines)
df['date'] = pd.date_range(start= df_date.Date.min(), periods=len(df), freq='D')

df_date.Date.min(),df_date.Date.max()
# # Exclude stopwords with Python's list comprehension and pandas.DataFrame.apply.
# df[0] = df[0].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

df = df.rename(columns = {0: 'text'})


# combine both
df=SP500_df.merge(df,on='date',validate='one_to_many')

df.head(2)
df.date.max()
del SP500_df
df.shape
df.head(20)


# lets keep what we need
trn=df[['date','ret_30D','text']]
# lets first drop some spam..easiest way to drop it is to just drop duplicates
org_len=trn.shape[0]
trn.drop_duplicates('text',inplace=True)
# Checking the length of amt of data is dropped
len(trn)/org_len
trn.head()


# lets just get a daily bag of words for the day
trn
trn.head(2)
trn['text']=trn[['date','ret_30D','text']].groupby(['date','ret_30D'])['text'].transform(lambda x: ' '.join(x))
trn.drop_duplicates(inplace=True)
trn.reset_index(drop=True)

#%% TFID


# ## Vectorize the text
# ml models cannot handle text. we need to convert it to numbers. easiest one is just a count vectorizer or a tfidf

from sklearn.feature_extraction.text import TfidfVectorizer
corpus=trn.text.tolist()


# keeping maximum features to 1000 to just limit the noise

#https://stackoverflow.com/questions/57359982/remove-stopwords-in-french-and-english-in-tfidfvectorizer
from nltk.corpus import stopwords
final_stopwords_list = stopwords.words('english')

vectorizer = TfidfVectorizer(max_features=50,ngram_range=(2,2),stop_words=final_stopwords_list)
X = vectorizer.fit_transform(corpus)
words=vectorizer.get_feature_names_out()
words[:10]
X.shape


# ## Fit matrix to returns
# 
# now we can fit this matrix to the returns. lets use a very simple linear regression model
from sklearn.linear_model import ElasticNet
regr = ElasticNet(random_state=0)
regr.fit(X, trn.ret_30D.values)
print(regr.intercept_)


# all the coeffs are 0 hmm lets make it easier for hte model to fit :)
np.sum(regr.coef_)
print(regr.coef_[:50])
regr = ElasticNet(random_state=0,alpha=0.001)
regr.fit(X, trn.ret_30D.values)
np.sum(regr.coef_)


# only a few words were pickde up.. lets see the words
coeff=pd.DataFrame(regr.coef_,columns=['value'])
coeff

# there are only 11 words picked up, wonder if it makes sense?

pos=coeff[coeff.value>0].index
[words[x] for x in pos]

neg=coeff[coeff.value<0].index
[words[x] for x in neg]


# ### remove urls
import re
trn.text=trn.text.apply(lambda text: re.sub(r"http\S+", "", text))
trn.text=trn.text.apply(lambda text: re.sub(r'https?://\S+|www\.\S+', '', text))

# ### remove html tags
CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

def cleanhtml(raw_html):
  cleantext = re.sub(CLEANR, '', raw_html)
  return cleantext

trn.text=trn.text.apply(lambda text: cleanhtml(text))


trn[trn.text.str.contains('d8')].sample(1).text.iloc[0][:100]
trn[trn.text.str.contains('[removed]')].sample(1).text.iloc[0][:100]
trn.text=trn.text.str.replace('[removed]','',regex=False)

#Return it back to corpus
corpus=trn.text.tolist()


def tfidf_elnet(max_features=15,ngram_range=(1,3),max_df=1, min_df=1, corpus=corpus,alpha=0.01):
    print(f'max_features:{max_features}, ngram_range:{ngram_range}, max_df:{max_df}, min_df:{min_df}, alpha:{alpha}')
    vectorizer = TfidfVectorizer(max_features=max_features,
                                 ngram_range=ngram_range,
                                 max_df=max_df,
                                 min_df=min_df,
                                 stop_words=final_stopwords_list)
    X = vectorizer.fit_transform(corpus)
    words=vectorizer.get_feature_names_out()

    regr = ElasticNet(random_state=0,alpha=alpha)
    regr.fit(X, trn.ret_30D.values)
    coeff=pd.DataFrame(regr.coef_,columns=['value'])
    
    if np.sum(regr.coef_)>0:
        pos=coeff[coeff.value>0].index
        print([words[x] for x in pos])
        neg=coeff[coeff.value<0].index
        print([words[x] for x in neg])
    else: print('all coeffs have weight 0')
    return regr


elnet=tfidf_elnet(alpha=0.001,max_df=0.8) #max_features are 1000 by default
elnet=tfidf_elnet(alpha=0.01,max_df=0.8,max_features=None)
elnet=tfidf_elnet(alpha=0.001,max_df=0.8,max_features=None)

elnet=tfidf_elnet(alpha=0.001,max_df=0.8,ngram_range=(0,1))
elnet=tfidf_elnet(alpha=0.001,max_df=0.8,ngram_range=(1,2))



elnet=tfidf_elnet(alpha=0.001,max_df=0.8,ngram_range=(2,2))
elnet=tfidf_elnet(alpha=0.001,max_df=0.8,ngram_range=(2,3))
elnet=tfidf_elnet(alpha=0.001,max_df=0.8,ngram_range=(3,4))

# ## future work: train with different horizons, different text datasets etc
# Will scalar or normalizating the prices have any changes, how will it affect elastic net?
# Different time horizons (lead and lag)












