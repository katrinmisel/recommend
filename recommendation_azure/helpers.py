# define the function to recommend new articles to a user
def recommend_articles(user_id, model, user_item_matrix):
    # get the row of the user in the user-item matrix
    user_row = user_item_matrix.getrow(user_id)
    # get the recommendations for the user
    recommendations = model.recommend(user_id, user_row, N = 5)
    # get the article ids from the recommendations
    return recommendations[0]