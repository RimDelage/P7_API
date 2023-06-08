# -*- coding: utf-8 -*-
import flask
from flask import Flask
from flask import request, jsonify
import pickle
import pandas as pd
import numpy as np

# # Reduce Memory Usage
# def reduce_mem_usage(df):
#     """ iterate through all the columns of a dataframe and modify the data type
#         to reduce memory usage.
#     """
#     start_mem = df.memory_usage().sum() / 1024 ** 2
#     #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
#
#     for col in df.columns:
#         col_type = df[col].dtype
#
#         if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
#             c_min = df[col].min()
#             c_max = df[col].max()
#             if str(col_type)[:3] == 'int':
#                 if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
#                     df[col] = df[col].astype(np.int8)
#                 elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
#                     df[col] = df[col].astype(np.int16)
#                 elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
#                     df[col] = df[col].astype(np.int32)
#                 elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
#                     df[col] = df[col].astype(np.int64)
#             else:
#                 if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
#                     df[col] = df[col].astype(np.float16)
#                 elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
#                     df[col] = df[col].astype(np.float32)
#                 else:
#                     df[col] = df[col].astype(np.float64)
#         elif 'datetime' not in col_type.name:
#             df[col] = df[col].astype('category')
#
#     #end_mem = df.memory_usage().sum() / 1024 ** 2
#     #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
#     #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
#
#     return df
#
#

# Initialisation de l'API
app = flask.Flask(__name__)
app.config["DEBUG"] = True

# Importer le modèle entrainé lightGBM
model = pickle.load(open('best_model_lgbm.pkl', 'rb'))


# Chargement du dataset
#data = reduce_mem_usage((pd.read_csv('df_1000.csv')))
data = pd.read_csv('df_1000.csv')
data.set_index('SK_ID_CURR', inplace = True)
data.drop(['TARGET','ypred1'], axis=1, inplace=True)


#extraction de la liste de clients
def list_id_clients():
    return data.index.tolist()

@app.route('/', methods=['GET'])
def Hello ():
    return 'test hello'


# @app.route('/predict/<id_client>', methods=['GET'])
# def get_predict(id_client:int):
#     index = data.index.get_loc(int(id_client))
#     #on récupère les features du client
#     features = data.iloc[index]
#      #on créer un disctionnaire pour la prédiction
#     result_pred =  {}
#     result_pred['predictions'] = model.predict_proba([features])[0,1].tolist()
#     return jsonify(result_pred)

def predict_client(id_client):
    index = data.index.get_loc(int(id_client))
    features = data.iloc[index]
    prediction = model.predict_proba([features])[0, 1].tolist()
    return prediction

@app.route('/predict/<id_client>', methods=['GET'])
def get_predict(id_client:int):
    prediction = predict_client(id_client)
    result_pred = {'predictions': prediction}
    return jsonify(result_pred)

if __name__ == "__main__":
     app.run(debug=True, port=4000)
