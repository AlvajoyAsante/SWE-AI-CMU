# Movie Recommendation System Report

## Learning

The goal of this project was to develop a movie recommendation system that can recommend movies based on user interaction history and available user and movie information.

### Data Used

The project uses three primary datasets:

- **Events Data:** Contains user interactions with movies, including ratings, movie IDs, user IDs, timestamps, and event types. Ratings range from 1 to 10 and are used to train the recommendation model.

- **Users Data:** Contains information about users, including demographic information and self-described preferences. The self-description fields are cleaned and tokenized during preprocessing.

- **Movies Data:** Contains movie metadata, including movie IDs, titles, genres, and overviews. Movie genres are converted into one-hot encoded features.

### Machine Learning Techniques

#### Collaborative Filtering

The primary recommendation technique used in this project is **Singular Value Decomposition (SVD)** from the Surprise library.

SVD uses historical user ratings to learn latent relationships between users and movies. The model is trained using the user ID, movie ID, and rating from the events dataset.

The model is evaluated using:

- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**

These metrics measure the difference between the ratings predicted by the model and the users' actual ratings.

#### Data Preprocessing and Feature Engineering

Several preprocessing techniques are applied to the datasets:

- Missing ratings are removed from the events data.
- Event timestamps are converted to datetime values.
- Event types are converted into binary indicator columns.
- User self-descriptions are converted to lowercase.
- Punctuation is removed from user self-descriptions.
- User self-descriptions are tokenized.
- User gender is converted into numerical values.
- Movie genres are converted into one-hot encoded features.
- Average user ratings are calculated from the training interactions.
- Movie interaction counts are calculated from the training data.

The processed user, movie, and event data are also merged to allow information from the different datasets to be analyzed together.

### Cold Start Problem

The cold start problem occurs when a recommendation system has little or no historical interaction data for a user.

For this project, user profile information, including self-described preferences and demographic information, is used as additional information that can help provide recommendations when a user has limited interaction history.

The system therefore does not rely exclusively on historical ratings. User and movie metadata are processed and incorporated into the recommendation pipeline to provide additional information for users with limited interaction history.

### Implementation

The primary implementation is contained in:

```text
src/train.py
```

The training process performs the following steps:

1. Loads the events, users, and movies datasets.
2. Cleans and preprocesses the datasets.
3. Splits the event data into training and testing sets.
4. Prepares the rating data for the Surprise library.
5. Trains an SVD collaborative filtering model.
6. Evaluates the model using RMSE and MAE.
7. Generates movie recommendations for a user.

---

## Running Your Model

### 1. Environment Setup

Make sure Python 3.x is installed.

Install the required dependencies:

```bash
pip install pandas scikit-learn scikit-surprise matplotlib
```

Clone the repository:

```bash
git clone <repository_url>
cd <repository_folder>
```

### 2. Data Preparation

Place the required datasets in the `data` directory:

```text
data/
├── events.csv.gz
├── users.csv.gz
└── movies.csv.gz
```

The datasets must contain the fields required by the training script, including user IDs, movie IDs, ratings, genres, and movie metadata.

### 3. Train the Model

Run the training script:

```bash
python src/train.py
```

The script will:

- Load the datasets.
- Preprocess the user, movie, and event data.
- Split the interaction data into training and testing sets.
- Train the SVD recommendation model.
- Calculate RMSE and MAE.
- Generate example movie recommendations.

### 4. Generate Recommendations

The current implementation generates recommendations directly from `train.py`.

The example recommendation call is:

```python
recommended_movies = recommend_based_on_description(user_id=1, n=10)
```

To generate recommendations for another user, change the `user_id`:

```python
recommended_movies = recommend_based_on_description(user_id=2, n=10)
```

The function returns the top 10 recommended movies with their movie IDs and titles.
