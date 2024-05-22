import os
import csv
import io
from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource # used for REST API building
from lstm.StockDataFrame import StockDataFrame
from lstm.LSTMClass import LSTMClass
from lstm.data_retrival import generate_data_for_one

# Change variable name and API name and prefix
stock_api = Blueprint('stock_api', __name__,
                   url_prefix='/api/stocks')

# API docs https://flask-restful.readthedocs.io/en/latest/api.html
api = Api(stock_api)

class StockAPI:     
    class Action(Resource):
        def get(self):
            ticker = request.args.get('ticker')  # get ticker argument from url
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            predictions_dir = os.path.abspath(os.path.join(curr_dir, '..', 'lstm', 'predictions'))
            json_ready =[]
            for filename in [x.split('.')[0] for x in os.listdir(predictions_dir)]:
                print(ticker, filename)
                if ticker == filename:
                    with open(predictions_dir + f"/{ticker}.csv", 'r') as file:
                        json_ready = [line.rstrip() for line in file]
                    return jsonify(json_ready)
                
            

            json_ready="File does not exist, pulling data..."
            generate_data_for_one(ticker)
            stock_df = StockDataFrame(ticker)
            stock_model = LSTMClass(ticker, stock_df.get_data_frame())
            stock_model.graph_predicted_data()
            with open(predictions_dir + f"/{ticker}.csv", 'r') as file:
                json_ready = [line.rstrip() for line in file]

            return jsonify(json_ready)  # jsonify creates Flask response object, more specific to APIs than json.dumps


    # building RESTapi endpoint, method distinguishes action
    api.add_resource(Action, '/')
