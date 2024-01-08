#!/usr/bin/env python
# coding: utf-8

# ## As a data scientist, you have been hired by a rookie movie producer to help him decide what type of movies to produce and which actors to cast. Your task is to analyze the data he has provided, which includes information on 3,000 movies, and use your findings to make recommendations.
# 
# ## To do this, you will first need to explore and clean the data to ensure that it is accurate and complete. This may involve checking for missing values, duplicate data, and outliers.
# 
# ## Once the data is clean, you can start to identify trends and patterns. For example, you could look at the most profitable movies, the most popular genres, and the actors who have starred in the most successful films.
# 
# ## You could also use the data to create predictive models. For example, you could develop a model that can predict the profitability of a movie based on its genre, budget, and cast.
# 
# ## Once you have a good understanding of the data, you can start to make recommendations to the movie producer. For example, you could recommend that he produce movies in certain genres or that he cast certain actors.
# 
# ## You could also provide him with more specific recommendations, such as suggesting that he produce a remake of a successful film or that he create a new franchise based on a popular book series.
# 
# ## Ultimately, the goal of your analysis is to help the movie producer make informed decisions about his business. By providing him with valuable insights and recommendations, you can help him to increase his chances of success.
# 
# ## Further, you have to answer the following questions:
# 1. ### <b> Which movie made the highest profit? Who were its producer and director? Identify the actors in that film.</b>
# 2. ### <b>This data has information about movies made in different languages. Which language has the highest average ROI (return on investment)? </b>
# 3. ### <b> Find out the unique genres of movies in this dataset.</b>
# 4. ### <b> Make a table of all the producers and directors of each movie. Find the top 3 producers who have produced movies with the highest average RoI? </b>
# 5. ### <b> Which actor has acted in the most number of movies? Deep dive into the movies, genres and profits corresponding to this actor. </b>
# 6. ### <b>Top 3 directors prefer which actors the most? </b>

# In[ ]:


#Import package
import pandas as pd
import numpy as np


# In[ ]:


imdb_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/imdb_data.csv')


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


imdb_df.head(2)


# In[ ]:


imdb_df.info()


# In[ ]:


print(imdb_df.columns)


# ## After reading all of the questions, identify the columns of data that are necessary to answer each question in order to obtain accurate insights. Keep all non-null columns.

# In[ ]:


columns_to_keep= ['budget', 'genres','original_language', 'original_title','cast', 'crew', 'revenue'] #Feature Selection


# In[ ]:


imdb_df = imdb_df[columns_to_keep]


# In[ ]:


print(imdb_df.columns)


# In[ ]:


#find all the row indexes for which genres is not null
imdb_df.loc[~imdb_df['genres'].isna(),'genres']


# In[ ]:


type(imdb_df.loc[0,'cast'])


# The eval() function in Python is a built-in function that evaluates a string as a Python expression. This means that you can use the eval() function to execute arbitrary Python code from a string.
# 
# 

# In[ ]:


expression = "1 + 2"
result = eval(expression)
print(result)


# In[ ]:


def convert_to_list(str):
  return eval(str)


# In[ ]:


#apply the above function only on non null values in genres column
imdb_df.loc[~imdb_df['genres'].isna(),'genres']= imdb_df.loc[~imdb_df['genres'].isna(),'genres'].apply(convert_to_list)


# In[ ]:


#apply the above function only on non null values in cast column
imdb_df.loc[~imdb_df['cast'].isna(),'cast']= imdb_df.loc[~imdb_df['cast'].isna(),'cast'].apply(convert_to_list)


# In[ ]:


#apply the above function only on non null values in crew column
imdb_df.loc[~imdb_df['crew'].isna(),'crew']= imdb_df.loc[~imdb_df['crew'].isna(),'crew'].apply(convert_to_list)


# In[ ]:


type(imdb_df.loc[0,'cast'])


# In[ ]:


imdb_df_new = imdb_df.copy()


# #Q1.Which movie made the highest profit? Who were its producer and director? Identify the actors in that film.

# In[ ]:


#checking for sanity in budget columns (outliers,vague values etc)
imdb_df_new.describe()
#budget of a movie in general cannot be 0 hence replacing those value with 0


# In[ ]:


imdb_df_new[imdb_df_new['budget']==0].head(3)


# In[ ]:


imdb_df_new['budget'].median()


# In[ ]:


#Replace extremely low values of budget and revenue column with median values of budget, revenue
imdb_df_new.loc[imdb_df_new['budget']<1000,'budget']= imdb_df_new['budget'].median()

imdb_df_new.loc[imdb_df_new['revenue']<1000,'revenue']= imdb_df_new['revenue'].median()



# In[ ]:


imdb_df_new.describe() #now fine


# In[ ]:


imdb_df_new['genres'].isnull().sum()


# In[ ]:


#create profit and ROI column
imdb_df_new['profit'] = imdb_df_new['revenue'] - imdb_df_new['budget']
imdb_df_new['roi']= 100* (imdb_df_new['profit']/imdb_df_new['budget'])


# In[ ]:


imdb_df_new.head(2)


# In[ ]:


#maximum profit
imdb_df_new['profit'].max()


# In[ ]:


#find index or row which have the max profit using .idxmax()
#.idxmax()-->> returns the row number(index) for the max value of the column
imdb_df_new['profit'].idxmax()


# ###The movie which made the highest profit is:

# In[ ]:


imdb_df_new.loc[imdb_df_new['profit'].idxmax(),'original_title']


# In[ ]:


max_profit_movie_df = imdb_df_new.iloc[imdb_df_new['profit'].idxmax()]


# In[ ]:


max_profit_movie_df.head()


# In[ ]:


max_profit_movie_df.loc['cast'][0]['name']


# In[ ]:


crew_list= max_profit_movie_df.loc['crew']
crew_list[0:3]


# In[ ]:


producer_list=[]
director_list=[]
for elem in crew_list:
 if elem['job']=='Producer':
   producer_list.append(elem['name'])
 if elem['job']=='Director':
   director_list.append(elem['name'])


# In[ ]:


print(f'PRODUCERS : {producer_list}')
print(f'DIRECTORS : {director_list}')


# In[ ]:


cast_list =max_profit_movie_df['cast']


# In[ ]:


cast_list[0:3]


# ###Actors in the Highest profit movie

# In[ ]:


actor_list=[]
for elem in cast_list:
  actor_list.append(elem['name'])


# In[ ]:


#actors
print(f'Actors of the movie are :')
actor_list


# #Q2.This data has information about movies made in different languages. Which language has the highest average ROI (return on investment)?

# In[ ]:


#we already calculated roi above
#df['roi'] = 100 * df['profit']/df['budget']


# In[ ]:


#Use groupby function on movie languages and ROI and finding mean
imdb_df_new.groupby('original_language')['roi'].mean().reset_index().sort_values(by='roi',ascending=False).head(3)


# In[ ]:


print('Language with highest average roi is')
imdb_df_new.groupby('original_language')['roi'].mean().reset_index().sort_values(by='roi',ascending=False).iloc[0,0]


# #Q3.Find out the unique genres of movies in this dataset.

# In[ ]:


#considering only those rows in genres column which have no null values
no_na_genres = imdb_df_new[~imdb_df_new['genres'].isna()]


# In[ ]:


type(no_na_genres)


# In[ ]:


no_na_genres.loc[0,'genres']


# In[ ]:


no_na_genres.loc[3,'genres'][0]


# In[ ]:


no_na_genres.loc[3,'genres']


# In[ ]:


#create a list of genres and using .iterrow() method to iterate over genres column
# .iterrow() --->> same as enumerate() its compulsory to use it in case of DataFrame
gen_list=[]
for index,row in no_na_genres.iterrows():
  genre = no_na_genres.loc[index,'genres']
  for k in genre:
    gen_list.append(k['name'])

#unique list of genres are:
pd.DataFrame(set(gen_list),columns=['Unique Genres'])


# In[ ]:


#considering only those rows in crew column which have no null values
no_na_crew = imdb_df_new[~imdb_df_new['crew'].isna()]


# In[ ]:


no_na_crew.shape


# In[ ]:


#A simple function extract list of all producers for a given movie_index
def create_producer_list(index):
  movie_index=no_na_crew.iloc[index]
  crew_list=movie_index.loc['crew']
  producer_list=[]
  for elem in crew_list:
    if elem['job']=='Producer':
        producer_list.append(elem['name'])
        return producer_list


# In[ ]:


create_producer_list(60)


# In[ ]:


#A simple function extract list of all director for a given movie_index
def create_Director_list(index):
  movie_index=no_na_crew.iloc[index]
  crew_list=movie_index.loc['crew']
  for elem in crew_list:
    if elem['job']=='Director':
      return elem['name']


# In[ ]:


create_Director_list(61)


# In[ ]:


# Create a empty dataframe with required column name in which we will append data later
Table = pd.DataFrame (columns=['Movie Title','Producers','Director','ROI'])


# In[ ]:


for index,row in no_na_crew.iterrows():

    try:
        Table = Table.append({'Movie Title':no_na_crew.loc[index,'original_title'],
                            'Producers':create_producer_list(index),
                            'Director':create_Director_list(index),
                            'ROI':no_na_crew.loc[index,'roi']},ignore_index=True)
    except:
        continue



# In[ ]:


Table.head(10)


# In[ ]:


#considering only those rows in cast column which have no null values
no_na_cast = imdb_df_new[~imdb_df_new['cast'].isna()]


# In[ ]:


no_na_cast.loc[0,'cast'][0]['name']


# In[ ]:


actor_list=[]
for index,row in no_na_cast.iterrows():
  for item in no_na_cast.loc[index,'cast']:
    if type(item)== dict:
      actor= item['name']
      actor_list.append(actor)


# In[ ]:


#create a  DataFrame with actor list
Actor_Table = pd.DataFrame(actor_list,columns=['Name of Actor'])


# In[ ]:


Actor_Table.shape


# In[ ]:


Actor_Table.head()


# In[ ]:


#sorting the actors using groupby function
Actor_Table.value_counts().reset_index().head()


# In[ ]:


print('Samuel L. Jackson and Robert De Niro both have done 30 films')


# In[ ]:


profit1=[]
profit2=[]
movie1=[]
movie2=[]
for index,row in no_na_cast.iterrows():
  for iter in no_na_cast.loc[index,'cast']:
    if type(iter)== dict:
      actor= iter['name']
      if 'Robert De Niro' in actor:
        profit1.append(no_na_cast.loc[index,'profit'])
        movie1.append(no_na_cast.loc[index,'original_title'])



# In[ ]:


if 'Samuel L. Jackson' in actor:
        profit2.append(no_na_cast.loc[index,'profit'])
        movie2.append(no_na_cast.loc[index,'original_title'])




# In[ ]:


#creating a loop to get the genres for Robert and Samuel
gener_r=[]
a=[]
for i in range(len(movie1)):
  for g in no_na_cast.loc[i,'genres']:
    a.append(g['name'])

  gener_r.append(a)
  a=[]

gener_s=[]
b=[]
for i in range(len(movie2)):
  for g in no_na_cast.loc[i,'genres']:
    b.append(g['name'])

  gener_s.append(b)
  b=[]



# In[ ]:


genr = np.array(gener_r)
gens = np.array(gener_s)


# In[ ]:


#creating sub dataframe for Robert
mov1= pd.DataFrame(movie1,columns=['Movie Name'])
prof1=pd.DataFrame(profit1,columns=['Movie Profit'])
gen1= pd.DataFrame(genr.flatten(),columns=['Genres'])


# In[ ]:


Movies_by_Robert=pd.concat([mov1,gen1,prof1],axis=1)


# In[ ]:


Movies_by_Robert.sort_values(by='Movie Profit',ascending=False).head()


# In[ ]:


#creating sub dataframe for Samuel
mov2= pd.DataFrame(movie2,columns=['Movie Name'])
prof2=pd.DataFrame(profit2,columns=['Movie Profit'])
gen2= pd.DataFrame(gens.flatten(),columns=['Genres'])


# In[ ]:


Movies_by_Samuel=pd.concat([mov1,gen1,prof1],axis=1)


# In[ ]:


Movies_by_Samuel


# In[ ]:




