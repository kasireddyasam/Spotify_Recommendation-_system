#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df= pd.read_csv("spotify_millsongdata.csv")


# In[3]:


df.head()


# In[4]:


df= pd.DataFrame(data=df,columns=df.columns)


# In[5]:


df.shape


# In[6]:


df.describe()


# # Data preprocessing(data cleaning)
# 

# ## step 1: Missing values
# ### Removing missing rows or fill data by its mean/midean/mode

# ## step 2: Remove duplicates
# ### in my data no duplicates.

# # step 3: Consistency
# ### Change all text data into Lower or Upper case

# # step 4: Remove irrelevent information
# ### In my data song link

# df = df.drop(columns=["link"])

# In[8]:


df["artist"] = df["artist"].str.lower()
df["song"]= df["song"].str.lower()
df["text"]= df["text"].str.lower()


# In[9]:


df.head()


# In[10]:


df["text"][1] 


# In[11]:


df["text"]= df["text"].replace(r'\r\n',' ',regex=True) 
df["text"]=df["text"].replace(r',','',regex=True)


# In[12]:


df["text"][1]


# # Normalization

# ## step 1:Tokenization 
# ### Splitinh text into indual words 

# ## step 2: Stop words Removal
# ###  Remove common words which doesn't have its own meaning 

# ## step 3: Stemming or Lemmatizer
# ## reduce tokens(words) to their root form 

# In[13]:


import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
print(stop_words)


# In[14]:


df = df.head(10000) # considering 10000 samples to analysis 
df.shape


# In[15]:


import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    # Tokenize and remove stop_words from the text
    tokens = nltk.word_tokenize(text)
    filtered_words = [word for word in tokens if word not in stop_words]
    # Lemmatize words (changing every word to its root once)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    return " ".join(lemmatized_words)

# Apply preprocessing function to the "text" column of your DataFrame
df["text"] = df["text"].apply(preprocess_text)


# In[ ]:





# # Vector Represntation of text data
# 
# ###### 1. Collecting all unique words (or tokens) from the entire dataset.
# ######  2. Counting the frequency of each word in each document.
# ###### 3. Representing each document as a vector, where each element of  the vector corresponds to the frequency of a unique word in the document. 

# In[16]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(df['text']).toarray()
print(X_tfidf)
print(X_tfidf.shape)
print(tfidf)


# ###### I would like to add an additional column to the DataFrame that indicates the mood of each song, such as whether it is a love song, sad song, romantic song, or pop song. Some songs may contain a mix of emotions, such as both love and sadness. Additionally, I want to create a separate DataFrame that displays the percentage of lyrics corresponding to each mood category. This will make it easier to search for songs by mood, instead of manually checking each row in the TF-IDF matrix. This approach will save time and streamline the process of finding songs with specific emotional themes. 

# ## # Step 1: Define Mood Categories

# In[18]:


mood_categories = ['love', 'sad', 'romantic', 'pop'] 

def categorize_sentiment(sentiment_polarity):
    if sentiment_polarity >= 0.6:
        return 'love'
    elif 0.2 <= sentiment_polarity < 0.6:
        return 'romantic'
    elif -0.2 <= sentiment_polarity < 0.2:
        return 'neutral'
    elif -0.6 <= sentiment_polarity < -0.2:
        return 'sad'
    else:
        return 'pop'




# ## Step 2: Analyze Lyrics (You can use sentiment analysis or other techniques here)

# In[19]:


from textblob import TextBlob
# define a function to analyze sentiment of each song
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity
    return categorize_sentiment(sentiment_polarity)

# Apply sentiment analysis to the lyrics column and
#create a new column for sentiment categories(love or sad or pop)

df["sentiment_category"] = df["text"].apply(analyze_sentiment)


# In[36]:


def percentage_sentiment(text):
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity
    return sentiment_polarity

df["%sentiment"] = df["text"].apply(percentage_sentiment)

df.rename(columns={'sentiment_category' : 'mood',"song" : "song_name"}, inplace=True)


    


# In[37]:


df.head(100)


# ## Input from user
# ### Implemented user-friendly search functionality allowing users to discover songs based on mood, song name, or artist name.  
# ### Leveraging optimized search techniques

# In[44]:


def preprocess_df(df):
    # Shorten DataFrame based on %sentiment
    df = df.sort_values(by='%sentiment')
    df = df.head(3000)  # Assuming you want to work with the top 3000 sentiment rows
    return df

def adjust_sentiment_for_sad(df):
    # Multiply %sentiment by -1 for rows where sentiment_category is "sad"(more the -ve value ->good rating one )
    df.loc[df['mood'] == 'sad', '%sentiment'] *= -1
    return df

def recommend_songs(df, input_criteria):
    # Preprocess DataFrame
    df = preprocess_df(df)
    # Adjust sentiment for "sad" mood
    df = adjust_sentiment_for_sad(df)
    
    # Case 1: If song name is given
    if 'song_name' in input_criteria:
        song_name = input_criteria['song_name']
        target_sentiment = df[df['song_name'] == song_name]['%sentiment'].iloc[0]
        recommended_songs = df.iloc[(df['%sentiment'] - target_sentiment).abs().argsort()[:10]]['song_name'].tolist()
        return recommended_songs
    
    # Case 2: If only mood is given
    elif 'mood' in input_criteria:
        mood = input_criteria['mood']
        mood_df = df[df['mood'] == mood]
        recommended_songs = mood_df.sort_values(by='%sentiment', ascending=False).head(10)['song_name'].tolist()
        return recommended_songs
    
    # Case 3: If artist name is given
    elif 'artist' in input_criteria:
        artist_name = input_criteria['artist']
        artist_df = df[df['artist'] == artist_name]
        recommended_songs = artist_df.sort_values(by='%sentiment', ascending=False).head(10)['song_name'].tolist()
        return recommended_songs
    
    else:
        return []


# In[66]:


# Example usage:
input_criteria = {
    # Choose one of the following options:
    # 'song_name': 'your_song_name_here',  # Case 1
    #'mood': 'your_mood_here',            # Case 2
    #'artist': 'your_artist_name_here'     # Case 3
    #"song_name" : "the name of the game"
    #"mood" : "sad"
    "artist" : "abba"
}

recommended_songs = recommend_songs(df, input_criteria)
print("Top 10 recommended songs:")
for idx, song in enumerate(recommended_songs, start=1):
    print(f"{idx}. {song}")



# In[ ]:




