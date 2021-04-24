import pandas as pd
import numpy as np
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt

movielens_data_file_url = ("http://files.grouplens.org/datasets/movielens/ml-latest-small.zip")

#### Downloads a file from a URL if it not already in the cache. ####
movielens_zipped_file = keras.utils.get_file("ml-latest-small.zip", movielens_data_file_url, extract = False)

#### Path to dataset in zip ####
keras_datasets_path = Path(movielens_zipped_file).parents[0]
movielens_dir = keras_datasets_path / "ml-latest-small"

# Only extract the data the first time the script is run.
if not movielens_dir.exists():
    with ZipFile(movielens_zipped_file, "r") as zip:
        # Extract files
        print("Extracting all the files now...")
        zip.extractall(path=keras_datasets_path)
        print("Done!")

ratings_file = movielens_dir / "ratings.csv"

df = pd.read_csv(ratings_file)
# print(df)

user_ids = df["userId"].unique().tolist()
# print(user_ids)

user2user_encoded = {x: i for i, x in enumerate(user_ids)}
# print(user2user_encoded)

userencoded2user = {i: x for i, x in enumerate(user_ids)}
# print(userencoded2user)

movie_ids = df["movieId"].unique().tolist()
# print(len(movie_ids), ': number of movies')

movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
# print(movie2movie_encoded)
# print(movie_encoded2movie)

df["user"] = df["userId"].map(user2user_encoded)
# print(df)

df["movie"] = df["movieId"].map(movie2movie_encoded)
# print(df)

num_users = len(user2user_encoded)

num_movies = len(movie_encoded2movie)
# print(num_users, num_movies)

df["rating"] = df["rating"].values.astype(np.float32)

# print(df)

# min and max ratings will be used to normalize the ratings
min_rating = min(df["rating"])
max_rating = max(df["rating"])

# print(min_rating, max_rating)
print(
    "Number of users: {}, Number of Movies: {}, Min rating: {}, Max rating: {}".format(
        num_users, num_movies, min_rating, max_rating
    )
)

#### generate a sample random row or column from the function caller data frame. ####
df = df.sample(frac=1, random_state=42)
# print(df)

x = df[["user", "movie"]].values
# print(x)

# Normalize the targets between 0 and 1. Makes it easy to train.

y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
# print(y)

# Training on 90% of the data and validating on 10% of the data.
train_indices = int(0.9 * df.shape[0])

x_train, x_test, y_train, y_test = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:],
)

EMBEDDING_SIZE = 50


class RecommenderNet(keras.Model):

    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
		
		# Size of the vocabulary, i.e. maximum integer index + 1
		# Dimension of the dense embedding
		# Initializer for the embeddings matrix
		# Regularizer function applied to the embeddings matrix
		
		#### FOR USER ####
        self.user_embedding = layers.Embedding(
            num_users,                              
            embedding_size,							
            embeddings_initializer="he_normal",		
            embeddings_regularizer=keras.regularizers.l2(1e-6), 
        )
		
        self.user_bias = layers.Embedding(num_users, 1)
		
		#### FOR MOVIES ####
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer = "he_normal",
            embeddings_regularizer = keras.regularizers.l2(1e-6),
        )
		
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        # Add all the components (including bias)
        x = dot_user_movie + user_bias + movie_bias
        # The sigmoid activation forces the rating to between 0 and 1
        return tf.nn.sigmoid(x)


model = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE)

model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=0.001))

history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 64,
    epochs = 5,
    verbose = 1,
    validation_data = (x_test, y_test),
)

###### Code to plot loss vs epoch ######
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

######### Read movies.csv #########
movie_df = pd.read_csv(movielens_dir / "movies.csv")

user_id = 6  # Type a user id 
# user_id = df.userId.sample(1).iloc[0]  # Let us get a user and see the top recommendations.
movies_watched_by_user = df[df.userId == user_id]

movies_not_watched = movie_df[~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)]["movieId"]

movies_not_watched = list(set(movies_not_watched).intersection(set(movie2movie_encoded.keys())))

movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]

user_encoder = user2user_encoded.get(user_id)

user_movie_array = np.hstack(([[user_encoder]] * len(movies_not_watched), movies_not_watched))

ratings = model.predict(user_movie_array).flatten()

top_ratings_indices = ratings.argsort()[-10:][::-1]

recommended_movie_ids = [movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices]

print("Showing recommendations for user: {}".format(user_id))
print("====" * 9)
print("Movies with high ratings from user")
print("----" * 8)
top_movies_user = (
    movies_watched_by_user.sort_values(by="rating", ascending=False)
    .head(5)
    .movieId.values
)
movie_df_rows = movie_df[movie_df["movieId"].isin(top_movies_user)]
for row in movie_df_rows.itertuples():
    print(row.title, ":", row.genres)

print("----" * 8)
print("Top 10 movie recommendations")
print("----" * 8)
recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]
for row in recommended_movies.itertuples():
    print(row.title, ":", row.genres)