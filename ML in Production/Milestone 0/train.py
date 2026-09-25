import pandas as pd
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import accuracy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load events data
events = pd.read_csv('data/events.csv.gz', compression='gzip')

# Handle missing ratings
events = events.dropna(subset=['rating'])

# Convert timestamp to datetime
events['timestamp'] = pd.to_datetime(events['timestamp'])

# Convert event_type to binary indicators
events = pd.get_dummies(events, columns=['event_type'])

# Check for any remaining missing values
print(events.isnull().sum())

# Split data for training and testing
train_events, test_events = train_test_split(events, test_size=0.2, random_state=42)  # This uses sklearn's train_test_split

# Load users data
users = pd.read_csv('data/users.csv.gz', compression='gzip')

# Preprocess text fields using simple NLP (tokenization and lowercasing)
# Use raw strings for regex
users['self_description_likes'] = users['self_description_likes'].fillna('').str.lower().str.replace(r'[^\w\s]', '')
users['self_description_dislikes'] = users['self_description_dislikes'].fillna('').str.lower().str.replace(r'[^\w\s]', '')

# Example of simple tokenization
users['self_description_likes_tokens'] = users['self_description_likes'].apply(lambda x: x.split())
users['self_description_dislikes_tokens'] = users['self_description_dislikes'].apply(lambda x: x.split())

# Encode categorical variables (example: gender)
users['gender'] = users['gender'].map({'M': 0, 'F': 1, 'O': 2, 'U': 3})

# Check for any remaining missing values
print(users.isnull().sum())

# Load movies data
movies = pd.read_csv('data/movies.csv.gz', compression='gzip')

# Process genres using one-hot encoding
movies = pd.concat([movies, movies['genres'].str.get_dummies(sep='|')], axis=1)

# Drop original genres column
movies = movies.drop(columns=['genres'])

# Check for any remaining missing values
print(movies.isnull().sum())

# Merge datasets on user_id and movie_id
merged_data = pd.merge(train_events, users, on='user_id')
merged_data = pd.merge(merged_data, movies, left_on='movie_id', right_on='movie_id')

# Example feature: average user rating
user_avg_rating = train_events.groupby('user_id')['rating'].mean().reset_index()
merged_data = pd.merge(merged_data, user_avg_rating, on='user_id', suffixes=('', '_avg'))

# Example feature: movie genre popularity
movie_genre_popularity = train_events.groupby('movie_id')['rating'].count().reset_index(name='genre_popularity')
merged_data = pd.merge(merged_data, movie_genre_popularity, on='movie_id')

# Check merged data
print(merged_data.head())

print("Model Training:")

# Prepare data for Surprise
reader = Reader(rating_scale=(1, 10))  # assuming the rating scale is from 1 to 10

# Load the event data for surprise (only using 'user_id', 'movie_id', and 'rating')
event_data = train_events[['user_id', 'movie_id', 'rating']]

# Convert the data to Surprise format
data = Dataset.load_from_df(event_data, reader)

# Split data into training and testing sets
trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)  # This uses Surprise's train_test_split

# Use SVD algorithm
algo = SVD()

# Train the algorithm on the training set
algo.fit(trainset)

# Evaluate the model on the test set
predictions = algo.test(testset)

# Calculate accuracy metrics
print('RMSE:', accuracy.rmse(predictions))
print('MAE:', accuracy.mae(predictions))

print("Training completed.")

print("Predicting User 1 Movie Recommendation ...")

# Example: Vectorize user self-descriptions
tfidf = TfidfVectorizer(stop_words='english')

# Assuming 'users' DataFrame contains 'self_description' column
user_tfidf_matrix = tfidf.fit_transform(users['self_description'])

# Example: Vectorize movie overviews
movie_tfidf_matrix = tfidf.fit_transform(movies['overview'])

# Compute cosine similarity between users and movies
cosine_sim = cosine_similarity(user_tfidf_matrix, movie_tfidf_matrix)

# Function to recommend movies based on user self-description similarity
def recommend_based_on_description(user_id, n=10):
    # Get the user's similarity scores with all movies
    user_idx = users[users['user_id'] == user_id].index[0]
    sim_scores = list(enumerate(cosine_sim[user_idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top N similar movies
    top_movie_indices = [i[0] for i in sim_scores[:n]]
    recommended_movies = movies.iloc[top_movie_indices][['movie_id', 'title']]

    return recommended_movies

# Example: Recommend movies for user with user_id = 1
recommended_movies = recommend_based_on_description(user_id=1, n=10)
print("Recommended movies based on user description:")
print(recommended_movies)