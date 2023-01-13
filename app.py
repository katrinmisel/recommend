import pickle
import streamlit as st
import pandas as pd
import numpy as np
import implicit
from scipy.sparse import coo_matrix, csr_matrix
from helpers import *

with open('als_model.pkl', 'rb') as f:
    model = pickle.load(f)

clicks = pd.read_csv('clicks.csv')
rows = clicks['user_id']
cols = clicks['click_article_id']
data = np.ones(len(clicks))
user_item_matrix = coo_matrix((data, (rows, cols))).tocsr()

# define the function to recommend new articles to a user
def recommend_articles(user_id, model, user_item_matrix, n_articles=5):
    # get the row of the user in the user-item matrix
    user_row = user_item_matrix.getrow(user_id)
    # get the recommendations for the user
    recommendations = model.recommend(user_id, user_row, N = 5)
    # get the article ids from the recommendations
    return recommendations[0]


st.image('content.PNG', width=500)
st.title('My Content Recommendation')

text = st.text_input(label='Enter a user ID:')

st.write("")

if st.button('Predict'):
    #
    recommendation = recommend_articles(int(text), model, user_item_matrix)
    recommendation = str(recommendation) 
    st.subheader('Recommended articles for user with ID {} are:'.format(text))
    st.subheader(recommendation)