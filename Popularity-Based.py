import pandas
movies = pandas.read_csv("movies.csv")
credits = pandas.read_csv("credits.csv")
ratings = pandas.read_csv("ratings.csv")

m = movies["vote_count"].quantile(0.95)
C = movies["vote_average"].mean()

def weighted_rating(df, m=m, C=C):
    R = df["vote_average"]
    v = df["vote_count"]
    wr = ((v / (v+m)) * R) + (m / (v+m) * C)
    return wr

movies["weighted_rating"] = movies.apply(weighted_rating, axis=1)

print(movies.sort_values("weighted_rating", ascending=False)[["title", "weighted_rating"]].head(10))