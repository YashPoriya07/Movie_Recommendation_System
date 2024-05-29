#!/usr/bin/env python
# coding: utf-8

# ## Content Based Movie Recommender System

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head()


# In[4]:


credits.head()


# In[5]:


movies.shape


# In[6]:


credits.shape


# In[7]:


movies=movies.merge(credits,on='title')


# In[8]:


movies.shape


# In[9]:


movies.head()


# In[10]:


# columns we will not drop: genre, id, keywords, title, overview, cast, crew

movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[11]:


movies.head()


# In[12]:


movies.isnull().sum()


# In[13]:


movies.dropna(inplace=True)


# In[14]:


movies.duplicated().sum()


# In[15]:


movies.iloc[0].genres #it's not in a proper form -> it is a list of dictionary ->we'll make it in proper format


# In[16]:


import ast
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        i['name']
        L.append(i['name'])
    return L


# In[17]:


movies['genres']=movies['genres'].apply(convert)


# In[18]:


movies.head()


# In[19]:


movies['keywords']=movies['keywords'].apply(convert)


# In[20]:


movies['keywords']


# In[21]:


movies['cast'][0] #we want the 1st three dictonaries for retreiving important actors real name and not character name


# In[22]:


def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter=counter+1
        else:
            break
    return L


# In[23]:


movies['cast']=movies['cast'].apply(convert3)


# In[24]:


movies.head() #columns genres,keywords,cast are sorted


# In[25]:


movies['crew'][0] #we want to retrieve the name whose job is director


# In[26]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L


# In[27]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[28]:


movies.head() #crew column sorted


# In[29]:


movies['overview'][0] #we will convert the overview (string) in List so we can concatenate it with other lists


# In[30]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[31]:


movies.head()


# In[32]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x]) 

#this function will remove the blank spaces so our model doesn't get cofused between
#for eg: Sam Worthington and Sam Mendes -> therefore making them a 
#single entity/tag SamWorthington SamMendes


# In[33]:


movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x]) 
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x]) 
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x]) 


# In[34]:


movies.head()


# In[35]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[36]:


movies.head()


# In[37]:


df=movies[['movie_id','title','tags']]


# In[38]:


df.head()


# In[39]:


df['tags']=df['tags'].apply(lambda x:" ".join(x))


# In[40]:


df['tags'][0]


# In[41]:


df['tags']=df['tags'].apply(lambda x:x.lower())


# In[42]:


df.head()


# In[43]:


df.shape


# ### Vectorization

# In[44]:


#since there are similar word such as actor actors  activities activity  loved loving love
#so we apply stemmig to such data so for danced,dance,dancing -> dance
import nltk


# In[45]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[46]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[47]:


#for example
stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver jamescameron')


# In[48]:


df['tags']=df['tags'].apply(stem)


# In[49]:


df['tags']


# In[50]:


df['tags'][0]


# In[51]:


df['tags'][1]


# #### now we want to compare the similarity of movies we need to calculate the similarity of the tags and since these data is in a textual format we will convert them to vectors
# #### so now when we apply text vectorization -> bag of words
# #### all tags are a vector in space and when a user selects a movie -> we recommend him a movie with closest tags movie

# In[52]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[53]:


vectors=cv.fit_transform(df['tags']).toarray()


# In[54]:


vectors


# In[55]:


vectors.shape


# In[56]:


vectors[0]


# In[57]:


cv.get_feature_names_out()


# In[58]:


#now we will calculate the cosine distance between the vectors
from sklearn.metrics.pairwise import cosine_similarity


# In[59]:


cosine_similarity(vectors).shape #total=4806 movies so 1st movie ka distance 4806 movies ke sath
#similarly 2nd movie ka 4806 movies ke saath distanc -> therefore shape is:-


# In[60]:


similarity=cosine_similarity(vectors)


# In[61]:


similarity[0]


# In[62]:


similarity[1]


# In[63]:


list(enumerate(similarity[0])) #enumerate function tuples me convert kardega distances ko
                               #toh jab hum sorting apply kare toh indexing kho na jaye     


# In[64]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6] 
#key bata raha hai 1st nahi 2nd no. ke basis pe sorting karna hai  
#and we only want 5 similar movies


# In[80]:


def recommend(movie):
    movie_index=df[df['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(df.iloc[i[0]].title)
    


# In[81]:


recommend('Avatar')


# In[82]:


recommend('Batman')


# In[68]:


import pickle


# In[69]:


df.to_dict()


# In[70]:


pickle.dump(df.to_dict(),open('movies_dict.pkl','wb'))


# In[71]:


df['title']


# In[72]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




