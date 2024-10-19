# import libraries
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')





# read data
data=pd.read_csv(r"C:\Users\ELFEKY\Desktop\projects\NLP\movies recommendation\data\movies_metadata.csv")
df=data.copy()





# data preprocessing
df.drop_duplicates(inplace=True)

# the aim of this is for resourses
df=df.iloc[:15000,:]





# Simple Recommender system
""" Build Simple Recommender System based on the metric below:

    weightedRating(WR) = ((v/v+m).R) + ((m/v+m).C)
    v >> is the number of votes for the movie. (vote_count)
    m >> is the min votes required to be listed in chart. (based on negative vote)
    R >> is the average rating of the movie. (vote_average)
    C >> is the mean vote across the the whole report. (calculate from data) """

# Use the recommendation score to recommend the top 15 movies

# Function that computes the weighted rating of each movie
C = df['vote_average'].mean()
M = df['vote_count'].quantile(0.90)

def weighted_rating(x, M=M, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+M) * R) + (M/(M+v) * C)

# Filter out all qualified movies into a new DataFrame
q_movies = df.copy().loc[df['vote_count'] >= M]
# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

top10=q_movies.title.head(10)

titles=list (df.title)

# Content based recommender systems
#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df['overview'] = df['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Compute the cosine Similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and movie titles
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['title'].iloc[movie_indices]



