import logging
import azure.functions as func
import json
from azure.storage.blob import BlobServiceClient
import pickle
from helpers import *
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import keys

connectionstring = keys.connectionstring
containername = keys.containername

def main(req: func.HttpRequest) -> func.HttpResponse:

    logging.info('Python HTTP trigger function processed a request.')

    user_id = req.params.get('user_id')

    if not user_id:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            user_id = req_body.get('user_id')

    if user_id:

        user_id = int(user_id)

        blob_service_client = BlobServiceClient.from_connection_string(connectionstring)
        container_client = blob_service_client.get_container_client(containername)
        blob_client = container_client.get_blob_client("clicks.csv")

        download_stream = blob_client.download_blob()
        clicks = pd.read_csv(download_stream)

        rows = clicks['user_id']
        cols = clicks['click_article_id']
        data = np.ones(len(clicks))

        blob_client = container_client.get_blob_client("als_model.pkl")
        model_bytes = blob_client.download_blob().readall()
        model = pickle.loads(model_bytes)

        user_item_matrix = coo_matrix((data, (rows, cols))).tocsr()
        recommendations = recommend_articles(user_id, model, user_item_matrix)
        recommendations = str(recommendations)
        
        return func.HttpResponse(json.dumps(recommendations))