import pandas
from surprise import Dataset, Reader
from surprise import SVD

ratings = pandas.read_csv("ratings.csv")[["userId", "movieId", "rating"]]
ratings.head()

reader = Reader(rating_scale=(1,5))
dataset = Dataset.load_from_df(ratings, reader)

trainset = dataset.build_full_trainset()

svd = SVD()

svd.fit(trainset)

print(svd.predict(1, 2, 3))