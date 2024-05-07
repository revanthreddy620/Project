#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd # use for data manipulation and analysis
import numpy as np # use for multi-dimensional array and matrix

import seaborn as sns # use for high-level interface for drawing attractive and informative statistical graphics 
import matplotlib.pyplot as plt # It provides an object-oriented API for embedding plots into applications
get_ipython().run_line_magic('matplotlib', 'inline')
# It sets the backend of matplotlib to the 'inline' backend:
import time # calculate time 

from sklearn.linear_model import LogisticRegression # algo use to predict good or bad
from sklearn.naive_bayes import MultinomialNB # nlp algo use to predict good or bad

from sklearn.model_selection import train_test_split # spliting the data between feature and target
from sklearn.metrics import classification_report # gives whole report about metrics (e.g, recall,precision,f1_score,c_m)
from sklearn.metrics import confusion_matrix # gives info about actual and predict
from nltk.tokenize import RegexpTokenizer # regexp tokenizers use to split words from text  
from nltk.stem.snowball import SnowballStemmer # stemmes words
from sklearn.feature_extraction.text import CountVectorizer # create sparse matrix of words using regexptokenizes  
from sklearn.pipeline import make_pipeline # use for combining all prerocessors techniuqes and algos

from PIL import Image # getting images in notebook
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator# creates words colud

from bs4 import BeautifulSoup # use for scraping the data from website
from selenium import webdriver # use for automation chrome 
import networkx as nx # for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.

import pickle# use to dump model 

import warnings # ignores pink warnings 
warnings.filterwarnings('ignore')


# In[2]:


dataset  =pd.read_csv("phishing_site_urls.csv")


# In[3]:


dataset.head()


# In[4]:


get_ipython().system('pip install selenium')


# In[5]:


dataset.tail()


# In[6]:


dataset.info()


# In[7]:


dataset.isnull().sum()


# In[8]:


#crete a dataframe of classes counts
label_counts=pd.DataFrame(dataset.Label.value_counts())


# In[9]:


tokenizer = RegexpTokenizer(r'[A-Za-z]+')


# In[10]:


dataset.URL[0]


# In[11]:


#This will be pull letter which matches to expression
tokenizer.tokenize(dataset.URL[0]) #Using first row


# In[12]:


print('Getting words tokenized...')
t0 = time.perf_counter()
dataset['text_tokenized'] = dataset.URL.map(lambda t: tokenizer.tokenize(t)) # doing with all rows
t1 = time.perf_counter() - t0
print('Time taken',t1,'sec')


# In[13]:


dataset.sample(5)


# In[14]:


stemmer = SnowballStemmer("english") #choose a language


# In[15]:


print('Getting words stemmed...')
t0 = time.perf_counter()
dataset['text_stemmed'] = dataset['text_tokenized'].map(lambda l: [stemmer.stem(word) for word in l])
t1 = time.perf_counter() - t0
print('Time taken',t1,'sec')


# In[16]:


dataset.sample(5)


# In[17]:


print('Getting joiningwords...')
t0 = time.perf_counter()
dataset['text_sent'] = dataset['text_stemmed'].map(lambda l: ' '.join(l))
t1 = time.perf_counter() - t0
print('Time taken',t1,'sec')


# In[18]:


#sliceing classes
bad_sites = dataset[dataset.Label == 'bad']
good_sites = dataset[dataset.Label == 'good']


# In[19]:


bad_sites.head()


# In[20]:


good_sites.head()


# In[21]:


get_ipython().system('pip install matplotlib')


# In[22]:


import matplotlib.pyplot as plt


# In[23]:


from wordcloud import STOPWORDS


# In[24]:


pip install wordcloud


# In[25]:


from wordcloud import WordCloud


# In[26]:


pip install wordcloud


# In[27]:


import matplotlib.pyplot as plt


# In[28]:


def plot_wordcloud(text, mask=None, max_words=400, max_font_size=120, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'com','http'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='white',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    mask = mask)
    wordcloud.generate(text)
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'green', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  


# In[29]:


data = good_sites.text_sent
data.reset_index(drop=True, inplace=True)


# In[30]:


common_text = str(data)
common_mask = np.array(Image.open('star.png'))
plot_wordcloud(common_text, common_mask, max_words=400, max_font_size=120, 
               title = 'Most common words use in good urls', title_size=15)


# In[31]:


data = bad_sites.text_sent
data.reset_index(drop=True, inplace=True)


# In[32]:


common_text = str(data)
common_mask = np.array(Image.open('comment.png'))
plot_wordcloud(common_text, common_mask, max_words=400, max_font_size=120, 
               title = 'Most common words use in bad urls', title_size=15)


# In[33]:


pip install selenium


# In[34]:


import selenium


# In[35]:


from selenium import webdriver

# Path to the Chrome WebDriver executable
webdriver_path = r'C:\Users\tejasri\Downloads\chromedriver.exe'

# Initialize Chrome WebDriver with the specified path
browser = webdriver.Chrome(executable_path=webdriver_path)


# In[36]:


browser = webdriver.Chrome(r"chromedriver.exe")


# In[37]:


list_urls = ['https://www.ezeephones.com/','https://www.ezeephones.com/about-us'] #here i have taken phishing sites 
links_with_text = []


# In[38]:


for url in list_urls:
    browser.get(url)
    soup = BeautifulSoup(browser.page_source,"html.parser")
    for line in soup.find_all('a'):
        href = line.get('href')
        links_with_text.append([url, href])


# In[39]:


df = pd.DataFrame(links_with_text, columns=["from", "to"])


# In[40]:


df.head()


# In[41]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[42]:


G = nx.from_pandas_edgelist(df, source='from', target='to', create_using=nx.DiGraph())
plt.show()


# In[43]:


GA = nx.from_pandas_edgelist(df, source="from", target="to")
nx.draw(GA, with_labels=False)
plt.show()


# In[44]:


#create cv object
cv = CountVectorizer()


# In[45]:


#help(CountVectorizer())


# In[46]:


feature = cv.fit_transform(dataset.text_sent) #transform all text which we tokenize and stemed


# In[47]:


feature[:5].toarray() # convert sparse matrix into array to print transformed features


# In[48]:


trainX, testX, trainY, testY = train_test_split(feature, dataset.Label)


# In[49]:


# create lr object
lr = LogisticRegression()


# In[50]:


lr.fit(trainX,trainY)


# In[51]:


lr.score(testX,testY)


# In[52]:


Scores_ml = {}
Scores_ml['Logistic Regression'] = np.round(lr.score(testX,testY),2)


# In[53]:


print('Training Accuracy :',lr.score(trainX,trainY))
print('Testing Accuracy :',lr.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(lr.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(lr.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")


# In[54]:


pipeline_ls = make_pipeline(CountVectorizer(tokenizer = RegexpTokenizer(r'[A-Za-z]+').tokenize,stop_words='english'), LogisticRegression())
##(r'\b(?:http|ftp)s?://\S*\w|\w+|[^\w\s]+') ([a-zA-Z]+)([0-9]+) -- these tolenizers giving me low accuray 


# In[55]:


trainX, testX, trainY, testY = train_test_split(dataset.URL, dataset.Label)


# In[56]:


pipeline_ls.fit(trainX,trainY)


# In[57]:


pipeline_ls.score(testX,testY) 


# In[58]:


print('Training Accuracy :',pipeline_ls.score(trainX,trainY))
print('Testing Accuracy :',pipeline_ls.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(pipeline_ls.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(pipeline_ls.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")


# In[59]:


pickle.dump(pipeline_ls,open('phishing.pkl','wb'))


# In[60]:


loaded_model = pickle.load(open('phishing.pkl', 'rb'))
result = loaded_model.score(testX,testY)
print(result)


# In[61]:


predict_bad = ['yeniik.com.tr/wp-admin/js/login.alibaba.com/login.jsp.php','fazan-pacir.rs/temp/libraries/ipad','tubemoviez.exe','svision-online.de/mgfi/administrator/components/com_babackup/classes/fx29id1.txt']
predict_good = ['youtube.com/','youtube.com/watch?v=qI0TQJI3vdU','retailhellunderground.com/','restorevisioncenters.com/html/technology.html']
loaded_model = pickle.load(open('phishing.pkl', 'rb'))
#predict_bad = vectorizers.transform(predict_bad)
# predict_good = vectorizer.transform(predict_good)
result = loaded_model.predict(predict_bad)
result2 = loaded_model.predict(predict_good)
print(result)
print("*"*30)
print(result2)


# In[62]:


predict_bad = ['yeniik.com.tr/wp-admin/js/login.alibaba.com/login.jsp.php']
loaded_model = pickle.load(open('phishing.pkl', 'rb'))
#predict_bad = vectorizers.transform(predict_bad)
result = loaded_model.predict(predict_bad)
print(result)


# In[63]:


predict_P = ['mlrit.ac.in/']
loaded_model = pickle.load(open('phishing.pkl', 'rb'))
#predict_bad = vectorizers.transform(predict_bad)
result = loaded_model.predict(predict_P)
print(result)


