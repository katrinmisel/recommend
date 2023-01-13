import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import implicit
from scipy.sparse import coo_matrix, csr_matrix
import pickle

clicks = pd.read_csv('clicks.csv')
articles = pd.read_csv('articles.csv')

def get_cats(list_articles):
    return list(articles[articles['article_id'].isin(list_articles)]['category_id'])

def get_click_data(clicks_dir):

    clicks_path = []

    clicks_path = clicks_path + sorted(
        [
            os.path.join(clicks_dir, fname) for fname in os.listdir(clicks_dir) if fname.endswith(".csv")
        ]
    )

    _li = []

    for filename in clicks_path:
        df = pd.read_csv(filename, index_col=None, header=0)
        _li.append(df)

    clicks = pd.concat(_li, axis=0, ignore_index=True)
    clicks.to_csv('clicks.csv', index=False)
    
    return clicks

def largest_values(arr: np.ndarray, n: int):
    # Sort the array in descending order
    sorted_array = np.sort(arr.ravel())[::-1]
    return sorted_array[:n]

def find_coordinates(array, value):
    for i in range(len(array)):
        for j in range(len(array[i])):
            if array[i][j] == value:
                return (i, j)
    return None

# function that takes as input user_id
# gets the embeddings of the read articles from click_article_id
# and returns a list of recommended article IDs

def recommend_user_articles(embedding_matrix, userId, n_recommendations):

    read_articles = clicks[clicks['user_id'] == userId]['click_article_id'].tolist()

    # Get the embeddings of the read articles
    read_embeddings = embedding_matrix[embedding_matrix.index.isin(read_articles)]

    # Get the embeddings of the unread articles
    unread_embeddings = embedding_matrix[~embedding_matrix.index.isin(read_articles)]

    # Calculate the similarity between the read and unread articles
    similarity = cosine_similarity(read_embeddings, unread_embeddings)

    # Find the top n recommendations
    top_n_values = largest_values(similarity, n_recommendations)
    
    # Find the article IDs corresponding to the top n recommendations
    top_n_article_ids = []
    for value in top_n_values:
        article_id = find_coordinates(similarity, value)[1]
        top_n_article_ids.append(unread_embeddings.index[article_id])
        
    return top_n_article_ids


# function that merges click data and retrains the Implicit model

def update_implicit_model(clicks_dir):

    clicks = get_click_data(clicks_dir)
    clicks = clicks.sort_values(by=['user_id'])[['user_id', 'click_article_id']]

    pd.DataFrame.to_csv(clicks, 'clicks.csv', ignore_index=True)

    # Create user-item matrix

    rows = clicks['user_id']
    cols = clicks['click_article_id']
    data = np.ones(len(clicks))

    user_item_matrix = coo_matrix((data, (rows, cols))).tocsr()

    # Load and train the model

    model = implicit.als.AlternatingLeastSquares()
    model.fit(user_item_matrix)

    # Save the updated model
    
    with open('als_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Model updated.")