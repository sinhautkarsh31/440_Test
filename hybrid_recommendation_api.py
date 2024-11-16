from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
import pandas as pd
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Load datasets (adjust paths as needed)
movies = pd.read_csv('movies_with_year.csv')  # Movies dataset with "year"
ratings = pd.read_csv('ratings.csv')  # Ratings dataset

# Content-Based Filtering setup
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


def get_cbf_recommendations(preferred_genres, oldest_year):
    # Filter movies by genres and year
    filtered_movies = movies[
        (movies['genres'].str.contains('|'.join(preferred_genres), case=False, na=False)) &
        (movies['year'] >= oldest_year)
    ]

    # If no movies match the filter, return an empty list
    if filtered_movies.empty:
        return []

    # Filter cosine similarity matrix to only include filtered movies
    filtered_indices = filtered_movies.index.tolist()
    filtered_cosine_sim = cosine_sim[filtered_indices][:, filtered_indices]

    # Calculate average similarity scores for filtered movies
    avg_sim_scores = filtered_cosine_sim.mean(axis=1)
    top_indices = avg_sim_scores.argsort()[::-1][:10]  # Top 10 recommendations

    # Return the recommended movies as a list of dictionaries
    return filtered_movies.iloc[top_indices][['title', 'genres', 'year']].to_dict(orient='records')


# Collaborative Filtering setup
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
svd = SVD()
svd.fit(trainset)


def get_cf_recommendations(user_id, oldest_year):
    # Filter movies based on the year condition
    filtered_movies = movies[movies['year'] >= oldest_year]

    # Get all unique movie IDs from the filtered movies
    all_movie_ids = filtered_movies['movieId'].unique()

    # Get movies rated by the user
    user_rated = ratings[ratings['userId'] == user_id]['movieId'].tolist()

    # Get unrated movies within the filtered movie IDs
    unrated_movies = [m for m in all_movie_ids if m not in user_rated]

    # Predict ratings for unrated movies
    predictions = []
    for movie_id in unrated_movies:
        try:
            prediction = svd.predict(user_id, movie_id)
            predictions.append(prediction)
        except Exception as e:
            print(f"Error predicting for movie_id {movie_id}: {e}")

    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_predictions = predictions[:10]  # Top 10 recommendations

    # Return movie titles, genres, and ratings
    recommendations = []
    for pred in top_predictions:
        movie_row = filtered_movies[filtered_movies['movieId'] == pred.iid]
        if not movie_row.empty:
            recommendations.append({
                "title": movie_row['title'].values[0],
                "genres": movie_row['genres'].values[0],
                "rating": pred.est
            })

    return recommendations


# API Endpoint for Hybrid Recommendations
@app.route('/hybrid-recommendations', methods=['POST'])
def hybrid_recommendations():
    try:
        # Parse the JSON payload
        data = request.json

        # Validate input
        if not all(k in data for k in ('userId', 'preferredGenres', 'oldestYear')):
            return jsonify({"error": "Invalid input"}), 400

        user_id = data['userId']
        preferred_genres = data['preferredGenres']
        oldest_year = data['oldestYear']

        # Get content-based recommendations
        cbf_recommendations = get_cbf_recommendations(preferred_genres, oldest_year)

        # Get collaborative filtering recommendations
        cf_recommendations = get_cf_recommendations(user_id, oldest_year)

        # Combine both sets of recommendations (e.g., hybrid approach)
        hybrid_recommendations = cbf_recommendations[:5] + cf_recommendations[:5]  # Adjust weights as needed

        return jsonify(hybrid_recommendations)

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
