
import logging
import azure.functions as func

import os
import glob
import pickle
import flask
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import json

# API setup
app = flask.Flask(__name__)
app.config["DEBUG"] = True

# This shows the use of WsgiMiddleware, which redirects the invocations to Flask handler
def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    """Each request is redirected to the WSGI handler.
    """
    logging.info('Python HTTP trigger function processed a request.')

    return func.WsgiMiddleware(app.wsgi_app).handle(req, context)



# Create URL of Main Page:  http://127.0.0.1:5000 IF locally
@app.route("/", methods=["GET"])
def home():
    return "<h1>FLASK CONTENT-BASED PRECALCULATION PREDICTION API</h1><p>ENDPOINT for Predictions is under the /predict url route.</p>"




# Create URL of Predict Endpoint:http://127.0.0.1:5000/predict IF locally
@app.route("/predict", methods=["POST"])
def predict_top_items():

    # Convert the JSON user id received from the request to Dictionary then integer
    data_as_dict = flask.request.get_json()
    uid = int(data_as_dict["id"])

    # Read-In Users Interactions Log files
    def read_in_user_intersactions(files_path):

        #file_list = glob.glob(os.path.join(files_path, "*.csv"))
        file_list = os.listdir(files_path)

        list_of_dataframes = []

        for file in file_list:

            df = pd.read_csv(os.path.join(files_path,file)) 

            # Step below to make Azure deployment not to give an error msg
            df = pd.DataFrame(df)                        
            list_of_dataframes.append(df)           
            
        df_combined = pd.concat(list_of_dataframes, axis=0, ignore_index=True) 
                
        return df_combined
    
    
        

    users_interactions_folder = os.path.join(os.getcwd(), "oc-p9-api-contentbased-precalculation-all-in-one/clicks")
    df_users_interactions = read_in_user_intersactions(users_interactions_folder)


    # Function to Find last article clicked by a all users
    def find_all_users_last_article_clicked(df):

        df_sorted = df.sort_values(by=['user_id', 'click_timestamp'], ascending=[True, False], inplace=False)

        # Keep only the rows with the maximum timestamp for each unique user_id
        df_result = df_sorted.groupby('user_id').first().reset_index()[["user_id", "click_article_id"]].rename(
            columns={"click_article_id": "last_click_article_id"})

        return df_result

    df_all_users_last_clicked_article = find_all_users_last_article_clicked(df_users_interactions) 
    # This step ONLY for All-in-on API: Otherwise store the table above in a Database and query it for user last article cliked
    #user_last_article_clicked = int(df_all_users_last_clicked_article[df_all_users_last_clicked_article["user_id"] == uid]["last_click_article_id"][uid])
    user_last_article_clicked = 14
    del(df_users_interactions)


    
    # Articles Embeddings Info
    def get_articles_embeddings_info():
         
        embeddings_folder = os.path.join(os.getcwd(), "oc-p9-api-contentbased-precalculation-all-in-one/embeddings_matrix")
        # Read embeddings matrix
        with open(os.path.join(embeddings_folder, "articles_embeddings.pkl"), "rb") as file:
            array_loaded = pickle.load(file)
        # df_no_ids = pd.DataFrame(array_loaded)

        # Articles metadata
        article_metada_folder = os.path.join(os.getcwd(), "oc-p9-api-contentbased-precalculation-all-in-one/articles_info")
        df_metadata = pd.read_csv(os.path.join(article_metada_folder, "articles_metadata.csv") )
        df_ids = df_metadata[["article_id"]]

        return array_loaded, df_ids

    embeddings_matrix, df_article_ids = get_articles_embeddings_info()
    #subset_of_articles_size = 1000
    subset_of_articles_size = 20
    embeddings_matrix = embeddings_matrix[:subset_of_articles_size]
    df_article_ids = df_article_ids.head(subset_of_articles_size)



    # Functions for COSINE Similarity
    def compute_pairwise_cosine_similarity(array_data):

        # Compute the pairwise cosine similarities using the cosine_similarity function
        pairwise_similarity = cosine_similarity(array_data)

        return pairwise_similarity

    df_similarity_scores = pd.DataFrame(compute_pairwise_cosine_similarity(embeddings_matrix))
    df_similarity_scores_with_ids = pd.concat([df_article_ids, df_similarity_scores], axis=1)
    del df_similarity_scores


    # Find top n largest similarity values for a given article
    def find_largest_values_and_indices_cpu(data_frame, n):

        result_df = pd.DataFrame()

        renamed_columns = {index: value for index, value in zip(data_frame.index, data_frame["article_id"])}
        data_frame = data_frame.rename(columns=renamed_columns)
        
        for column in data_frame.columns:
            # Sort the column in descending order and get the top n values
            largest_values = data_frame.nlargest(n, column)

            # Get the row indices of the largest values
            row_indices = largest_values.index
            similar_articles_ids = data_frame["article_id"][row_indices].values

            # Create a temporary DataFrame with the largest values and row indices
            temp_df = pd.DataFrame(
                {
                    column: largest_values[column].values,
                    str(column) + "_similar_articles_ids": similar_articles_ids,
                    # str(column) + "_Row_Index": row_indices,
                }
            )

            # Concatenate the temporary DataFrame with the result DataFrame
            result_df = pd.concat([result_df, temp_df], axis=1)

        result_df = result_df.drop(0).drop(
            columns=[ "article_id", "article_id_similar_articles_ids",
                      # "article_id_Row_Index",
                     ]
              )

        return result_df
    

    top_values_to_keep = 6
    top_similarity_and_ids_embeddings = find_largest_values_and_indices_cpu(df_similarity_scores_with_ids, n=top_values_to_keep)

    col1 = str(user_last_article_clicked) + "_similar_articles_ids"
    top_n_values = top_similarity_and_ids_embeddings[[col1, user_last_article_clicked]].rename(columns={col1:"article_id", user_last_article_clicked: "similarity_score"})

    json_string = top_n_values.to_json()

    return flask.jsonify(response=json_string, message=user_last_article_clicked)