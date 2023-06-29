import sklearn
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

movies = pandas.read_csv("movies.csv", sep=",") 

merged = movies['overview'].str.cat([movies['keywords'], movies['genres']], sep=" ")
merged.to_csv('merged.csv', index=False)
merged = pandas.read_csv('merged.csv')

tfidf = TfidfVectorizer(stop_words="english")
merged['overview'] = movies['overview'].fillna("")
tfidf_matrix = tfidf.fit_transform(merged['overview'])

similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

def similar_movies(movie_title, nr_movies):
    idx = movies.loc[movies["title"]==movie_title].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    movies_indices = [tpl[0] for tpl in scores[1:nr_movies+1]]
    similar_titles = list(movies["title"].iloc[movies_indices])
    return similar_titles

print(similar_movies("Cars", 5))